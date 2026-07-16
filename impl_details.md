# device-oob-detector 实现细节与完整调用流程

本文档基于源码梳理这个「基于 CUDA Compute Sanitizer API 的 device 全局内存越界（OOB）检测工具」的完整调用流程、数据结构关系，以及两个核心机制（Patching 插桩、Shadow Table 合法区间表管理）的实现细节。

---

## 一、整体架构

核心思想：Host 侧维护一张「合法内存区间表」，通过 Sanitizer 把 device 补丁插桩到用户 kernel 的每一次全局访存指令上，访问前查表判断是否越界。

三层组成：

| 层 | 文件 | 职责 |
|---|---|---|
| **前端（两种接入方式）** | `src/oob_detector.cpp`（嵌入式）/ `src/external_inject.cpp`（外挂 LD_PRELOAD） | 把引擎挂进目标进程 |
| **Host 引擎（共享）** | `src/sanitizer_engine.cpp` | 封装 Callback / Patching / Memory 三套 API |
| **Device 补丁** | `device/oob_patch.cu` → `oob_patch.cubin` | 每次全局访存前查表、记录越界 |

`include/oob_detector.h` 是 host/device **共享的数据结构契约**（`OobAllocEntry` / `OobReport` / `OobDeviceCtx` 布局必须两边逐字节一致）。

---

## 二、启动阶段（Subscribe 必须先于任何 CUDA 调用）

两种接入方式殊途同归，最终都调用 `oobEngineInitEx()`。

### 方式1 嵌入式

用户在 `main()` 第一行调用 `oobStart(&cfg)`：

```c
OobConfig cfg = OOB_CONFIG_DEFAULT;
cfg.abortOnError = 0;
cfg.verbose = 1;
oobStart(&cfg);   /* must precede any CUDA call */
```

`oobStart` → `oobEngineInitEx`。

### 方式2 外挂（无需改/重编用户代码）

`tools/oob-check` 脚本设置环境变量后 `exec` 目标程序：

```bash
export CUDA_INJECTION64_PATH="$INJECT_SO"
export OOB_PATCH_CUBIN="$PATCH_CUBIN"
...
exec "$@"
```

CUDA 驱动在建 context 前 `dlopen` 该 so 并调用 `InitializeInjection()` → `oobEngineInitEx`。这是 Compute Sanitizer 官方注入点，保证订阅早于任何 CUDA 调用。

### `oobEngineInitEx` 做三件事

1. `sanitizerSubscribe(&subscriber, oobCallback, ...)` —— 注册唯一主回调。
2. `sanitizerEnableDomain(RESOURCE)` —— 捕获 module 加载 + alloc/free。
3. `sanitizerEnableDomain(LAUNCH)` —— 捕获 kernel begin/end。

之后所有事件都进入统一入口 `oobCallback`，按 domain 分发到 `onResource` / `onLaunch`。

> **"subscribe first" 铁律**：订阅必须早于第一个 CUDA context 创建，否则会漏掉早期 alloc 和 module 加载，导致区间表 / 插桩不完整。两种前端都是为满足这一点而设计的。

---

## 三、运行期回调流程

### A. RESOURCE 域 —— `onResource()`

处理三类事件：

1. **`MODULE_LOADED`**（模块加载）→ 插桩（详见第五节 Patching 机制）：
   - `ensurePatchesLoaded(ctx)`：`sanitizerAddPatchesFromFile(cubin, ctx)` 把补丁 cubin 加载进用户 context（每 context 一次）。
   - `sanitizerPatchInstructions(GLOBAL_MEMORY_ACCESS, module, "oob_global_access_cb")`：声明所有全局访存都调用补丁函数。
   - `sanitizerPatchModule(module)`：真正改写 SASS 插入回调。

2. **`DEVICE_MEMORY_ALLOC`** → `liveAllocs[base] = size`（登记合法区间）。

3. **`DEVICE_MEMORY_FREE`** → `liveAllocs.erase(base)`（这样访问已释放内存也会被判越界）。

### B. LAUNCH 域 —— kernel 前后

**`onLaunchBegin()`**（kernel 启动前）：

1. `ensureBuffers(ctx, nAllocs)`：用 `sanitizerAlloc`（非 `cudaMalloc`，因回调内不能调 runtime）分配三块 device buffer：`OobDeviceCtx` / `OobReport[]` / `OobAllocEntry[]`。
2. 把 `liveAllocs` 快照成 host 数组，`sanitizerMemcpyHostToDeviceAsync` 上传合法区间表（挂在 kernel 同一 stream 上，保证先于 kernel 落地）。
3. `sanitizerMemset` 清零 report buffer。
4. 填充 `OobDeviceCtx`（allocBase / reportBase / maxReports / abortOnError）并上传。
5. `sanitizerSetLaunchCallbackData(hLaunch, function, hStream, ctxDev)` —— 把这个 ctx 绑定到本次 launch，作为补丁的 `userdata`。

**Kernel 执行时 —— device 补丁 `oob_global_access_cb`**：

每次全局访存前被调用：

- `ctx` / `allocBase` 为空 → 返回 `SUCCESS`（引擎未就绪，绝不干扰用户 kernel）。
- `oob_addr_is_legal()` 线性扫描区间表，判断 `[ptr, ptr+accessSize)` 是否完全落在某合法区间内。
- 合法 → `SUCCESS`；否则 `atomicAdd(&reportCursor,1)` 抢一个槽位记录越界（pc / 地址 / 宽度 / 读写 / block / thread）。
- 若 `abortOnError` → 返回 `SANITIZER_PATCH_ERROR` 终止该 warp。

`reportCursor` 用 atomic 计数**尝试总数**（可超过 `maxReports`），buffer 只存前 `maxReports` 条。

**`onLaunchEnd()`**（kernel 结束后）：

1. `sanitizerStreamSynchronize` 等 device 写完。
2. `sanitizerMemcpyDeviceToHost` 拷回 `OobDeviceCtx` 读 `reportCursor`（= 尝试次数）。
3. 拷回有效的 `OobReport[]`，`totalErrors += attempted`。
4. `printReports()`：格式化打印，并从 host 的 `liveAllocs` 找最近区间算出 "overflow by N bytes"。

---

## 四、结束阶段 —— `oobEngineShutdown()`

- 嵌入式：用户调 `oobStop()`。
- 外挂：`InitializeInjection` 里 `atexit(onExit)` 注册，进程退出自动调用。

动作：`sanitizerUnsubscribe` 解除订阅 → 打印汇总 `Summary: N out-of-bounds access(es) across M kernel launch(es)`。

---

## 五、调用流程时序图

```
[启动]
 oobStart / InitializeInjection
   └─ oobEngineInitEx
        ├─ sanitizerSubscribe(oobCallback)
        └─ sanitizerEnableDomain(RESOURCE, LAUNCH)

[运行 — 事件驱动，全部经 oobCallback 分发]
 RESOURCE:
   MODULE_LOADED  ─ onResource ─ addPatchesFromFile → patchInstructions(GLOBAL) → patchModule
   MEM_ALLOC/FREE ─ onResource ─ 更新 liveAllocs 区间表
 LAUNCH:
   LAUNCH_BEGIN ─ onLaunchBegin ─ sanitizerAlloc/Memcpy/Memset 上传区间表+ctx
                                 → sanitizerSetLaunchCallbackData(ctx→userdata)
        │
        ▼   (kernel 在 device 上运行，每次全局访存)
   [device] oob_global_access_cb → oob_addr_is_legal? → 否则 atomicAdd 记录越界
        │
        ▼
   LAUNCH_END ─ onLaunchEnd ─ StreamSync → MemcpyD2H(ctx,reports) → printReports

[结束]
 oobStop / atexit(onExit)
   └─ oobEngineShutdown ─ sanitizerUnsubscribe → 打印 Summary
```

---

## 六、数据结构关系图（Host ↔ Device 内存布局）

核心是 `OobDeviceCtx`：它是 host 引擎和 device 补丁之间的唯一「契约」，通过 `sanitizerSetLaunchCallbackData` 作为 `userdata` 传给补丁。

```
HOST 侧 (sanitizer_engine.cpp)                    DEVICE 侧 (显存, 由 sanitizerAlloc 分配)
──────────────────────────────                    ─────────────────────────────────────

Engine g {
  std::map<uint64_t,uint64_t> liveAllocs   ┐
    base → size  (权威区间表)                │ onLaunchBegin 快照+上传
                                            │ sanitizerMemcpyHostToDeviceAsync
  std::map<CUcontext,PerLaunchBuffers> bufs │
    每 context 一组:                         ▼
      ┌─────────────────┐            ┌──────────────────────────┐
      │ PerLaunchBuffers│            │ OobAllocEntry[ allocCap ] │◄─┐  b.allocDev
      │  allocDev  ─────┼───────────▶│  {base,size}              │  │
      │  reportDev ─────┼──────┐     │  {base,size} ...          │  │
      │  ctxDev    ─────┼──┐   │     └──────────────────────────┘  │
      │  allocCap       │  │   │                                    │
      └─────────────────┘  │   │     ┌──────────────────────────┐  │
                           │   └────▶│ OobReport[ maxReports ]   │◄┐│  b.reportDev
  totalErrors (atomic)     │         │  {pc,addr,size,flags,     │ ││
  launchCount              │         │   block/thread xyz}       │ ││
}                          │         └──────────────────────────┘ ││
                           │                                       ││
                           │         ┌──────────────────────────┐ ││  b.ctxDev
                           └────────▶│ OobDeviceCtx              │ ││
                                     │   allocBase ──────────────┼─┘│ (指向 AllocEntry[])
                                     │   numAllocs               │  │
                                     │   reportBase ─────────────┼──┘ (指向 Report[])
                                     │   maxReports              │
                                     │   reportCursor (atomic)   │◄── device atomicAdd
                                     │   abortOnError            │
                                     └──────────────────────────┘
                                              ▲
                                              │ sanitizerSetLaunchCallbackData
                                              │ 绑定到本次 launch
                                              │
                              [device] oob_global_access_cb(userdata = &OobDeviceCtx)
```

指针链路：补丁只拿到一个 `userdata`（= `ctxDev`），通过 `ctx->allocBase` 找到区间表、`ctx->reportBase` 找到报告环，全部靠 `OobDeviceCtx` 里的两个 device 指针串联。这就是为什么 `oob_detector.h` 与 `oob_patch.cu` 的三个 struct **布局必须逐字节一致**（含 `_pad0/_pad1` 对齐填充）。

```c
typedef struct OobDeviceCtx {
    uint64_t allocBase;      /* device pointer to OobAllocEntry[numAllocs] */
    uint32_t numAllocs;
    uint32_t _pad0;
    uint64_t reportBase;     /* device pointer to OobReport[maxReports]    */
    uint32_t maxReports;
    uint32_t reportCursor;   /* atomicAdd cursor; total attempts (may > max) */
    uint32_t abortOnError;
    uint32_t _pad1;
} OobDeviceCtx;
```

---

## 七、深入 Patching 机制

Patching 是把「我们的检测代码」塞进「用户 kernel 的 SASS」的过程，分三步，全部发生在 `MODULE_LOADED` 回调里。

### 步骤1：加载补丁到 context —— `sanitizerAddPatchesFromFile`

- 补丁 `oob_patch.cu` 用特殊标志编译成 cubin：

```bash
nvcc --cubin --compile-as-tools-patch oob_patch.cu -o oob_patch.cubin -arch=sm_80 ...
```

  `--compile-as-tools-patch` 生成的不是普通 kernel，而是「可被 splice 的补丁函数」。
- 补丁与用户代码不在同一编译单元，所以必须显式加载到**用户代码所在的同一个 CUDA context**。因此用 `ctxPatched` 按 `CUcontext` 记账，每 context 只加载一次（`Add/Patch` 系列要求串行访问，故持有 `g.mtx`）。

### 步骤2：选择插桩点 —— `sanitizerPatchInstructions`

告诉 sanitizer：「本 module 里每一条 `GLOBAL_MEMORY_ACCESS`（全局 load/store/atomic）指令，都在执行前调用 device 函数 `oob_global_access_cb`」。这里只选 GLOBAL（不选 shared/local）是为了聚焦最常见的全局越界并降低开销。

### 步骤3：实际改写 SASS —— `sanitizerPatchModule`

前两步只是声明，`sanitizerPatchModule` 才真正把回调 splice 进 module 的机器码。**时机关键**：必须在 module 加载后、其 kernel 首次 launch 前完成 —— `MODULE_LOADED` 正好是这个 hook；若拖到 launch 时做会与首次 launch 竞态。

### 补丁函数签名（sanitizer 约定）

```c
oob_global_access_cb(void* userdata,
                     uint64_t pc,
                     void* ptr,
                     uint32_t accessSize,
                     uint32_t flags,
                     const void* /*pData*/)
```

`userdata` 由 `sanitizerSetLaunchCallbackData` 注入；`pc / ptr / accessSize / flags` 由 sanitizer 运行时在每次访存点自动填入。关键：补丁运行在实际访存**之前**，所以能在非法访问真正发生前拦截。

---

## 八、深入 Shadow Table（合法区间表）管理

这张表回答唯一的问题：「被访问的地址是否合法？」它的生命周期横跨 alloc/free 事件与每次 launch。

### 权威源：host 侧 `liveAllocs`

`std::map<base, size>`，由 RESOURCE 回调实时增删。free 后立即删除，因此**访问已释放内存（use-after-free 的地址）也会被判越界**。

### 每次 launch 前：快照 → 上传

device 不能直接读 host 的 `std::map`，所以每次 launch 把当前 live 区间「拍平」成数组，再 `sanitizerMemcpyHostToDeviceAsync` 挂在 **kernel 同一条 stream** 上，保证区间表先于 kernel 落地。

### buffer 复用策略：`ensureBuffers`

不是每次都重分配，而是按 context 缓存一组 buffer，仅当区间数超容量时才扩容（带 64 个 headroom），减少回调内分配开销。扩容时先 `sanitizerFree` 旧 buffer 再 `sanitizerAlloc` 新 buffer。

### device 侧查表：线性扫描

判定标准是 `[addr, addr+size)` **完全包含**在某区间内（`addr >= b && addr+size <= e`），所以跨越区间边界的部分访问也算越界。alloc 数通常很少（几百个以内），线性扫描足够。

### 越界报告的 "overflow by N" 是 host 侧算的

device 只记录裸事件（pc / 地址 / 宽度），host 打印时再回查 `liveAllocs` 找最近区间、算出溢出字节数。

---

## 九、其它值得注意的实现细节

- **atomic 计数超容量语义**：`reportCursor` 用 `atomicAdd` 递增，即使超过 `maxReports` 也继续计数 —— buffer 只存前 N 条，host 用 `attempted > count` 提示「还有多少条未显示」（`OOB_MAX_REPORTS` 可调）。
- **补丁的防御式返回**：当 `userdata` 或 `allocBase` 为空（launch 数据还没绑定的窗口期），补丁直接返回 `SUCCESS`，绝不干扰用户 kernel。
- **abortOnError 两处联动**：device 补丁返回 `SANITIZER_PATCH_ERROR` 终止越界 warp，host `onLaunchEnd` 读完报告后再 `abort()` 进程。
- **回调内禁用 CUDA runtime**：引擎在回调里一律使用 Memory API（`sanitizerAlloc` / `sanitizerMemcpy*` / `sanitizerMemset` / `sanitizerFree`），而非 `cudaMalloc` / `cudaMemcpy`，因为在 sanitizer 回调中调用 CUDA runtime 不受支持、可能死锁。
- **三套 Sanitizer API 各司其职**：Callback（订阅事件）、Patching（插桩 device 代码）、Memory（回调内安全分配/拷贝）。

---

## 十、涉及的关键 Sanitizer API 汇总

| 分类 | API | 用途 |
|------|-----|------|
| Callback | `sanitizerSubscribe` / `sanitizerUnsubscribe` | 注册/注销唯一主回调 |
| Callback | `sanitizerEnableDomain` | 开启 RESOURCE / LAUNCH 域 |
| Callback | `sanitizerSetLaunchCallbackData` | 把 `OobDeviceCtx` 绑定到本次 launch（作为补丁 userdata）|
| Patching | `sanitizerAddPatchesFromFile` | 把补丁 cubin 加载进 context |
| Patching | `sanitizerPatchInstructions` | 声明 GLOBAL 访存插桩点 |
| Patching | `sanitizerPatchModule` | 实际改写 module SASS |
| Memory | `sanitizerAlloc` / `sanitizerFree` | 回调安全的 device 分配/释放 |
| Memory | `sanitizerMemcpyHostToDeviceAsync` / `sanitizerMemcpyDeviceToHost` | 上传区间表+ctx / 回拷报告 |
| Memory | `sanitizerMemset` | 清零 report buffer |
| Memory | `sanitizerStreamSynchronize` | 读回前等待 device 写完 |
