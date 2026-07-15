# history.md — device-oob-detector 开发记录

## 改动摘要

| 时间 | commit | 摘要 |
|------|--------|------|
| 2026-07-15 | (init) | 初始化仓库结构 |
| 2026-07-15 | `见下` | 完成检测框架：共享引擎 + 嵌入式 API + 外挂注入 + device 补丁 + 样例 + Makefile，本机编译通过 |

> 本机（arm64/sbsa）**无 GPU**，无法做 device 端实跑；已完成全部**编译验证**，并确认 device 补丁符号 `oob_global_access_cb` 正确导出。带 GPU 环境按“测试用例”一节运行即可看到越界报告。

---

## 需求上下文

用户提供 `/home/Cuda-13/compute-sanitizer`（含 `include/` 与 `docs/`），要求做 **device 内存越界检测**，两种方式：
1. 在用户算子 host 入口调用 sanitizer API 初始化检测框架；
2. 外挂工具方式，开发一个二进制/库对通用算子做检测。
并给出方案、代码、接口说明与目的。

## 关键设计决策

1. **共用一套引擎**：`sanitizer_engine.cpp` 封装 Callback/Patching/Memory 三套 API，两个前端（嵌入式、外挂）都只是把引擎“挂进目标进程”的不同姿势。
2. **外挂用官方注入点 `CUDA_INJECTION64_PATH`**：驱动在建 context 前 `dlopen` 该 so 并调用 `InitializeInjection()`，这是 Compute Sanitizer 自身用的机制，保证订阅早于任何 CUDA 调用（比 `LD_PRELOAD` 更可靠，因此 `oob-check` 采用它）。
3. **检测粒度 = 全局访存**：patch `SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS`，回调签名用 `SanitizerCallbackMemoryAccess`。
4. **合法区间表来自 RESOURCE 回调**：`DEVICE_MEMORY_ALLOC/FREE` 维护 `base->size` map；每次 launch 前快照进 device buffer。
5. **device 端不可调 cudaMalloc**：所有 tracking buffer 用 Memory API（`sanitizerAlloc/sanitizerMemset/sanitizerMemcpy*`）。
6. **报告回传**：`LAUNCH_END` 里 `sanitizerStreamSynchronize` 后把 report buffer 拷回 host 格式化打印，并给出“越界多少字节”提示。

## 改动详情（时间线）

### 2026-07-15 —— 阅读头文件，确认 API 形态
读 `sanitizer.h / _memory.h / _patching.h / _callbacks.h / _result.h / _stream.h` 与 `SanitizerApiGuide`。确认：
- 订阅：`sanitizerSubscribe` + `sanitizerEnableDomain(RESOURCE|LAUNCH)`。
- patch 流程：`sanitizerAddPatchesFromFile` → `sanitizerPatchInstructions` → `sanitizerPatchModule` → `sanitizerSetLaunchCallbackData`。
- 补丁编译：`nvcc --cubin --compile-as-tools-patch`。
- 内存访问补丁签名 `SanitizerCallbackMemoryAccess(userdata,pc,ptr,accessSize,flags,pData)`。
- alloc 回调数据结构 `Sanitizer_ResourceMemoryData{address,size,...}`；launch 数据 `Sanitizer_LaunchData{hLaunch,function,hStream,functionName,...}`。

### 2026-07-15 —— 编写代码
新增文件：
- `include/oob_detector.h`：对外 C API + host/device 共享结构（`OobAllocEntry/OobReport/OobDeviceCtx`）。
- `device/oob_patch.cu`：device 补丁，线性扫描合法区间，越界写 report ring（原子游标）。
- `src/sanitizer_engine.cpp`：核心引擎。
- `src/oob_detector.cpp`：嵌入式前端（`oobStart/oobStop/...`）。
- `src/external_inject.cpp`：外挂前端（`InitializeInjection`，读环境变量配置）。
- `examples/vec_add_oob.cu`：故意 `i < n+extra` 越界样例。
- `tools/oob-check`：外挂启动脚本。
- `Makefile`。

### 2026-07-15 —— 编译修复
1. `readlink` 未声明 → 加 `#include <unistd.h>`。
2. 删除引擎里未使用/冲突的前置声明（`oobEngineInit(const EngineConfig*)`）。
3. `-lcuda` 找不到 → Makefile 增加 stub 路径 `lib64/stubs` 与 `targets/sbsa-linux/lib/stubs`。

`make` 全绿；`cuobjdump -symbols build/oob_patch.cubin` 可见 `oob_global_access_cb`（STB_GLOBAL）。

## 最终文件结构

```
device-oob-detector/
├── README.md
├── history.md
├── Makefile
├── include/oob_detector.h
├── src/sanitizer_engine.cpp
├── src/oob_detector.cpp
├── src/external_inject.cpp
├── device/oob_patch.cu
├── tools/oob-check
└── examples/vec_add_oob.cu
```

产物（`build/`）：`oob_patch.cubin`、`liboob_detector.so`、`liboob_inject.so`、`vec_add_oob`、`vec_add_oob_embedded`。

## 算法流程

```
subscribe(RESOURCE, LAUNCH)
 │
 ├─ RESOURCE.MODULE_LOADED  → addPatchesFromFile(cubin) → patchInstructions(GLOBAL_ACCESS) → patchModule
 ├─ RESOURCE.MEM_ALLOC      → liveAllocs[base] = size
 ├─ RESOURCE.MEM_FREE       → liveAllocs.erase(base)
 │
 ├─ LAUNCH_BEGIN
 │     snapshot liveAllocs → device allocTable
 │     memset reportBuffer = 0
 │     fill OobDeviceCtx{allocBase,numAllocs,reportBase,maxReports,abortOnError}
 │     sanitizerSetLaunchCallbackData(launch, ctxDev)
 │
 │   [device] 每次全局访存:
 │     oob_global_access_cb(ptr, size, flags):
 │        if [ptr,ptr+size) ⊄ 任一合法区间:
 │           idx = atomicAdd(reportCursor)
 │           if idx < max: 写 OobReport{pc,addr,size,flags,block/thread}
 │           if abortOnError: return PATCH_ERROR
 │
 └─ LAUNCH_END
       streamSynchronize
       copy ctx back → reportCursor
       copy reports back → 打印越界详情 + “overflow by N bytes”
```

## 测试用例汇总

| 用例 | 命令 | 期望 |
|------|------|------|
| 编译验证（本机，无 GPU）| `make` | 全部产物生成，无报错 ✅ |
| 补丁符号导出 | `cuobjdump -symbols build/oob_patch.cubin \| grep oob_global` | 见 `oob_global_access_cb` ✅ |
| 嵌入式检测（需 GPU）| `LD_LIBRARY_PATH=build:$SANITIZER_DIR OOB_PATCH_CUBIN=build/oob_patch.cubin ./build/vec_add_oob_embedded` | 报告 vecAddBad 对 `c[]` 的越界写（i∈[256,264)），末尾 `detected N OOB accesses` |
| 外挂检测（需 GPU，不改用户代码）| `tools/oob-check ./build/vec_add_oob` | 同上越界报告 + Summary |
| 首次越界即终止 | `tools/oob-check --abort ./build/vec_add_oob` | 打印后 `abort()` |
| 无越界回归 | 把样例 `extra=0` 重编运行 | 无报告，Summary=0 |

> 有 GPU 的机器上，若驱动/CUDA 版本与 Compute Sanitizer 不匹配，API 会返回 `SANITIZER_ERROR_NOT_INITIALIZED`（引擎会打印并禁用检测，不影响目标程序）。
