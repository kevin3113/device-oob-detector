# device-oob-detector — CUDA Device 内存越界检测框架

基于 **NVIDIA Compute Sanitizer API**（`/home/Cuda-13/compute-sanitizer`）实现的 device 端全局内存越界（Out-Of-Bounds, OOB）检测框架。

框架同时提供 **两种使用方式**：

| 方式 | 名称 | 适用场景 | 是否需要改用户代码 |
|------|------|----------|-------------------|
| 方式 1 | **嵌入式初始化 (embedded)** | 用户自己写算子/程序，愿意在 host 入口调一行初始化 | 需要（1 行）|
| 方式 2 | **外挂工具 (external / LD_PRELOAD)** | 对任意已编译好的通用算子二进制做检测，不改也不重编用户代码 | 不需要 |

两种方式**共用同一套检测引擎**（`sanitizer_engine`）与**同一份 device 补丁**（`oob_patch.cu`）。区别只在“如何把引擎挂进目标进程”。

---

## 1. 原理

Compute Sanitizer API 由三部分组成，本框架全部用到：

1. **Callback API**（`sanitizerSubscribe` / `sanitizerEnableDomain`）
   - 订阅 **RESOURCE** 域：捕获 `cudaMalloc/cudaFree`（device memory alloc/free）→ 维护一张“合法内存区间表”（shadow allocation table）。
   - 订阅 **RESOURCE / MODULE_LOADED**：模块加载时把 device 补丁 patch 进去。
   - 订阅 **LAUNCH** 域：kernel 启动前把本次 launch 的补丁回调数据（指向 device 端 report buffer）绑定到 launch；kernel 结束后把 report buffer 拷回 host 检查并打印越界报告。

2. **Patching API**（`sanitizerAddPatches*` / `sanitizerPatchInstructions` / `sanitizerPatchModule`）
   - 把编译好的 device 补丁 `oob_patch.cubin` 插入用户模块。
   - 对 `SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS`（全局内存读/写/原子）插桩，每次访问都会调用我们的 device 函数 `oob_global_access_cb`。

3. **Memory API**（`sanitizerAlloc` / `sanitizerMemcpyDeviceToHost` / `sanitizerMemset`）
   - 在 callback 内安全地分配 / 清零 / 回拷 device 端 report buffer（不能在 callback 里直接调 `cudaMalloc`）。

### 检测判定逻辑

- Host 侧持有全部合法区间 `[base, base+size)`（来自 alloc 回调）。
- Kernel 启动前，把“合法区间数组”通过 Memory API 写入一块 device buffer；device 补丁在每次全局访问时，用被访问地址 `ptr` 和访问宽度 `accessSize` 去查这张表：
  - 若 `[ptr, ptr+accessSize)` **完全落在某个合法区间内** → 放行。
  - 否则 → 记录一条越界事件（pc、地址、宽度、读/写、blockIdx/threadIdx）到 report buffer 的原子游标。
- Kernel 结束后 host 把 report buffer 拷回，格式化打印越界报告。

> 注：本框架聚焦 **global memory OOB**（最常见的算子越界）。同样的骨架可扩展到 shared/local（把 `SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS` 等一并 patch）。

---

## 2. 目录结构

```
device-oob-detector/
├── README.md                 # 本文件：方案 + 接口说明
├── history.md                # 开发时间线 / 改动记录
├── include/
│   └── oob_detector.h        # 对外 C API（嵌入式方式使用）+ 共享数据结构
├── src/
│   ├── sanitizer_engine.cpp  # 检测引擎（Callback/Patching/Memory API 全部封装）
│   ├── oob_detector.cpp      # 方式1：嵌入式 C API 实现（oobStart/oobStop）
│   └── external_inject.cpp   # 方式2：外挂 .so 的构造/析构入口（LD_PRELOAD）
├── device/
│   └── oob_patch.cu          # device 端补丁（被编译成 cubin，插桩全局访存）
├── tools/
│   └── oob-check             # 外挂启动脚本（设置 LD_PRELOAD 后 exec 目标程序）
├── examples/
│   └── vec_add_oob.cu        # 故意越界的样例算子，用于验证两种方式
└── Makefile
```

---

## 3. 构建

需要：CUDA 11.0+（本仓库基于 `/home/Cuda-13`，CUDA 13.3）、`nvcc`、Compute Sanitizer 头文件与 `libsanitizer-public.so`。

```bash
# 一次性设置（按你的机器路径）
export CUDA_HOME=/home/Cuda-13
export SANITIZER_DIR=$CUDA_HOME/compute-sanitizer
export PATH=$CUDA_HOME/bin:$PATH

make                 # 编译全部：libsanitizer 引擎 / 外挂 so / device cubin / 样例
make SM=sm_90        # 指定目标架构（默认 sm_80）
```

产物：
- `build/oob_patch.cubin`     device 补丁（两种方式都要用，路径由环境变量 `OOB_PATCH_CUBIN` 指定，默认取可执行文件同目录）
- `build/liboob_detector.so`  嵌入式库（方式1，链接用）
- `build/liboob_inject.so`    外挂注入库（方式2，LD_PRELOAD）
- `build/vec_add_oob`         样例程序

---

## 4. 使用接口

### 方式 1：嵌入式（在算子 host 入口初始化）

在你的算子/程序 host 代码里包含头文件并调用两行 API：

```c
#include "oob_detector.h"

int main() {
    OobConfig cfg = OOB_CONFIG_DEFAULT;
    cfg.patchCubinPath = "build/oob_patch.cubin"; // 可选，默认自动查找
    cfg.abortOnError   = 0;                       // 1=首次越界即终止
    oobStart(&cfg);          // ★ 必须在任何 CUDA 调用/建 context 之前调用

    // ... 你的正常 CUDA 代码：cudaMalloc / kernel<<<>>> / cudaMemcpy ...

    oobStop();               // 结束检测，打印汇总
    return 0;
}
```

编译链接：

```bash
nvcc your_op.cu -o your_op \
     -I device-oob-detector/include \
     -L device-oob-detector/build -loob_detector \
     -I $SANITIZER_DIR/include -L $SANITIZER_DIR -lsanitizer-public
# 运行前保证能找到 so 和 cubin
export LD_LIBRARY_PATH=device-oob-detector/build:$SANITIZER_DIR:$LD_LIBRARY_PATH
export OOB_PATCH_CUBIN=device-oob-detector/build/oob_patch.cubin
./your_op
```

#### 嵌入式 C API（`include/oob_detector.h`）

| 函数 | 说明 |
|------|------|
| `int  oobStart(const OobConfig* cfg)` | 初始化并订阅 sanitizer。**务必在创建 CUDA context / 第一个 CUDA API 之前调用**。返回 0 成功。|
| `void oobStop(void)` | 反订阅、拷回并打印剩余报告、释放资源、输出统计汇总。|
| `unsigned long long oobErrorCount(void)` | 返回到目前为止累计检测到的越界次数。|
| `void oobSetAbortOnError(int enable)` | 运行期切换“首次越界即 `abort()`”。|

`OobConfig` 字段：

| 字段 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `patchCubinPath` | `const char*` | `NULL` | device 补丁路径；NULL 时依次查 `$OOB_PATCH_CUBIN`、可执行文件同目录 `oob_patch.cubin` |
| `abortOnError` | `int` | 0 | 检测到越界是否立即终止 |
| `verbose` | `int` | 1 | 是否打印每条越界详情（0 只打印汇总）|
| `maxReports` | `int` | 4096 | 单次 kernel 最多记录多少条越界（device buffer 容量）|

### 方式 2：外挂工具（不改用户代码）

对**任意**用 CUDA 写的通用算子二进制，直接用启动器包一层：

```bash
device-oob-detector/tools/oob-check ./your_prebuilt_op arg1 arg2 ...
```

`oob-check` 做的事：
1. 设置 `CUDA_INJECTION64_PATH=<绝对路径>/liboob_inject.so`
   （CUDA 驱动的官方注入点：进程启动、建 context 前，驱动会 `dlopen` 该 so 并调用其 `InitializeInjection()`。这正是 Compute Sanitizer 自身使用的机制，保证在任何 CUDA 调用前订阅成功。）
2. 设置 `OOB_PATCH_CUBIN` 指向 `oob_patch.cubin`。
3. `exec` 目标程序。

也可手动：

```bash
export CUDA_INJECTION64_PATH=$PWD/build/liboob_inject.so
export OOB_PATCH_CUBIN=$PWD/build/oob_patch.cubin
export LD_LIBRARY_PATH=$SANITIZER_DIR:$LD_LIBRARY_PATH
./your_prebuilt_op
```

环境变量开关（方式2）：

| 变量 | 默认 | 含义 |
|------|------|------|
| `OOB_PATCH_CUBIN` | 同目录 | device 补丁 cubin 路径 |
| `OOB_ABORT_ON_ERROR` | 0 | 1=首次越界 abort |
| `OOB_VERBOSE` | 1 | 每条越界是否详打 |
| `OOB_MAX_REPORTS` | 4096 | device report buffer 容量 |

---

## 5. 越界报告格式

```
========== [OOB] Out-of-bounds GLOBAL access detected ==========
  kernel   : vecAddBad
  access   : WRITE  size=4 bytes
  address  : 0x7f0a1c000400   (nearest alloc: base=0x7f0a1c000000 size=1024, overflow by 4 bytes past end)
  thread   : block=(3,0,0) thread=(96,0,0)
  pc       : 0x5f0
================================================================
```

结束时打印：

```
[OOB] Summary: 12 out-of-bounds access(es) across 5 kernel launch(es).
```

---

## 6. 目的与价值

- **及早发现算子 bug**：global 越界读/写往往表现为偶发精度异常或随机崩溃，用本框架可精确定位到 kernel、线程、访问地址与越界字节数。
- **两种接入姿势**：
  - 开发期用**嵌入式**，与单测/CI 集成，越界即失败。
  - 对**已交付的通用算子二进制**（无源码/不便重编）用**外挂**方式做回归扫描。
- **对齐官方机制**：外挂使用 CUDA 官方 `CUDA_INJECTION64_PATH` 注入点，嵌入式使用同一引擎，行为一致、可维护。

详见 `history.md` 的算法流程与测试用例。
