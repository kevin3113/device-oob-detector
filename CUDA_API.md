# device-oob-detector 使用的 CUDA / Sanitizer API 梳理

本文档梳理 `device-oob-detector`（CUDA 设备端全局内存越界检测器）在实现中用到的
所有 CUDA 相关 API，按功能分类整理，并说明**每个 API 在本项目中的具体作用**。

项目的核心思路：基于 **NVIDIA Compute Sanitizer API** 在**模块加载时**把一段设备补丁
（device patch）拼接进用户 kernel，让每次 global load/store/atomic **在真正访问之前**先经过
补丁回调 `oob_global_access_cb`，与 host 侧维护的"合法地址区间表"比对，越界即记录上报。

支持两种模式（共享同一套引擎 `sanitizer_engine.cpp`）：
- **嵌入式**（`oobStart`/`oobStop`，改用户代码）；
- **外挂式**（`CUDA_INJECTION64_PATH` 注入 `.so`，不改、不重编）。

---

## 1. Compute Sanitizer Callback API（订阅 + 回调分发）

> 头文件 `sanitizer.h`；实现见 `src/sanitizer_engine.cpp`。
> 这是整个检测器的骨架：通过订阅 RESOURCE / LAUNCH 两个回调域驱动全部逻辑。

| API | 分类 | 在本项目中的作用 |
|-----|------|------------------|
| `sanitizerSubscribe(&subscriber, oobCallback, userdata)` | 订阅 | 注册唯一的主回调 `oobCallback`。**必须在首个 CUDA 调用/首个 context 之前**完成，否则会漏掉早期的分配与模块加载事件。全局只能有一个订阅者。 |
| `sanitizerUnsubscribe(subscriber)` | 退订 | shutdown 时释放订阅者槽位、停止后续所有回调，避免回调进入已释放的引擎状态。 |
| `sanitizerEnableDomain(1, subscriber, SANITIZER_CB_DOMAIN_RESOURCE)` | 域开关 | 启用 RESOURCE 域：接收模块加载（用于打补丁）与设备内存 alloc/free（用于维护合法区间表）。 |
| `sanitizerEnableDomain(1, subscriber, SANITIZER_CB_DOMAIN_LAUNCH)` | 域开关 | 启用 LAUNCH 域：接收每次 kernel 启动的 begin/end 钩子，用于绑定回调数据、回收上报。只启这两个域以降低回调开销。 |
| `sanitizerGetResultString(result, &str)` | 错误处理 | 把 `SanitizerResult` 错误码转可读字符串，用于 `SAN_CHECK` 宏诊断。 |

### 相关回调数据结构 / 常量

| 结构体 / 常量 | 作用 |
|---------------|------|
| `Sanitizer_ResourceModuleData` | 模块加载回调载荷，`->context`、`->module` 用于打补丁。 |
| `Sanitizer_ResourceMemoryData` | 内存 alloc/free 回调载荷，`->address`、`->size` 用于维护合法区间表。 |
| `Sanitizer_LaunchData` | 启动回调载荷：`->context`、`->module/function`、`->hLaunch`、`->hStream`、`->functionName`。 |
| `SANITIZER_CBID_RESOURCE_MODULE_LOADED` | 模块加载事件——打补丁的精确时机（加载后、kernel 运行前）。 |
| `SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC` / `_FREE` | 设备内存分配/释放事件——增删合法区间。 |
| `SANITIZER_CBID_LAUNCH_BEGIN` / `_END` | kernel 启动前/后事件。 |

---

## 2. Compute Sanitizer Patching API（把补丁拼进用户 kernel）

> 头文件 `sanitizer.h`（host 侧）；实现见 `src/sanitizer_engine.cpp`。
> 补丁本体是 `device/oob_patch.cu` 编成的 cubin，需要在**每个 context**加载并注入一次。

| API | 分类 | 在本项目中的作用 |
|-----|------|------------------|
| `sanitizerAddPatchesFromFile(cubinPath, ctx)` | 加载补丁 | 把独立编译的 `oob_patch.cubin`（`nvcc --compile-as-tools-patch` 产物）加载进用户代码所在的**同一 CUDA context**。补丁是 per-context 的，故按 CUcontext 记录、每 ctx 只加载一次。 |
| `sanitizerPatchInstructions(SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "oob_global_access_cb")` | 选择插桩点 | 声明"本模块内每个 global load/store/atomic 都调用设备函数 `oob_global_access_cb`"。只选 GLOBAL（不含 shared/local）以聚焦全局越界、降低开销。 |
| `sanitizerPatchModule(module)` | 应用插桩 | 真正改写模块 SASS，把选定回调插入进去。必须在 `PatchInstructions` 之后、kernel 运行之前调用。 |

---

## 3. Compute Sanitizer Memory API（回调安全的设备内存操作）

> 头文件 `sanitizer.h`；实现见 `src/sanitizer_engine.cpp`。
> **关键约束**：在 sanitizer 回调内部**不能**调用 CUDA Runtime（`cudaMalloc`/`cudaMemcpy` 等），
> 否则可能死锁。因此所有回调内的设备内存操作都用 Sanitizer 的回调安全等价物。

| API | 分类 | 在本项目中的作用 |
|-----|------|------------------|
| `sanitizerAlloc(ctx, &ptr, bytes)` | 分配 | 回调内分配设备内存：`OobDeviceCtx` 控制块、`OobReport[]` 上报环、`OobAllocEntry[]` 合法区间表。是 `cudaMalloc` 的回调安全等价物。 |
| `sanitizerFree(ctx, ptr)` | 释放 | 当合法区间表需要扩容时，先释放旧缓冲区。`cudaFree` 的回调安全等价物。 |
| `sanitizerMemset(ptr, 0, bytes, hStream)` | 清零 | 在 LAUNCH_BEGIN 时清零本次上报环，避免上次 launch 的计数泄漏；按 kernel 所在 stream 定序。`cudaMemset` 的回调安全等价物。 |
| `sanitizerMemcpyHostToDeviceAsync(dst, src, bytes, hStream)` | 上传 | 在 LAUNCH_BEGIN 时把 host 侧的合法区间表快照 + `OobDeviceCtx` 控制块推到设备，压在与 kernel 相同的 stream 上，保证补丁读取前已就位。 |
| `sanitizerMemcpyDeviceToHost(dst, src, bytes, hStream)` | 回读 | 在 LAUNCH_END 时先回读 `OobDeviceCtx`（拿到 `reportCursor` = 越界尝试总数），再回读有效的 `OobReport[]` 记录用于打印。 |
| `sanitizerStreamSynchronize(hStream)` | 同步 | LAUNCH_END 回读前等待 kernel 及其补丁回调把上报写完、对 host 可见。`cudaStreamSynchronize` 的回调安全等价物。 |

---

## 4. Compute Sanitizer Launch API（把控制块绑定到本次启动）

> 头文件 `sanitizer.h`；实现见 `src/sanitizer_engine.cpp`。

| API | 分类 | 在本项目中的作用 |
|-----|------|------------------|
| `sanitizerSetLaunchCallbackData(hLaunch, function, hStream, ctxDevPtr)` | 绑定 | 把本次 launch 专属的 `OobDeviceCtx` 设备指针绑定到**这一次**启动，使补丁回调 `oob_global_access_cb` 运行时能通过 `userdata` 拿到它。**per-launch**（区别于 per-function 的 `sanitizerSetCallbackData`）——每次 launch 都用刚重置过的上报环 + 当刻的分配快照。 |

---

## 5. Compute Sanitizer 设备补丁 API（device 侧）

> 头文件 `sanitizer_patching.h`；实现见 `device/oob_patch.cu`。
> 补丁函数在每个 global 访问**之前**被调用，是实际做越界判断与记录的地方。

| 符号 / 返回值 | 作用 |
|---------------|------|
| `oob_global_access_cb(userdata, pc, ptr, accessSize, flags, pData)` | 补丁回调本体（`extern "C" __device__ SanitizerPatchResult`）。接收访问地址/大小/读写标志，与合法区间表比对；越界则 `atomicAdd` 占槽并写入 `OobReport`。 |
| `SANITIZER_PATCH_SUCCESS` | 返回值：允许访问照常进行（合法、或引擎未就绪时的安全放行）。 |
| `SANITIZER_PATCH_ERROR` | 返回值：`abortOnError` 时返回，令 sanitizer 退出该越界线程所在的整个 warp，及时止损（host 仍能读回上报）。 |
| `SanitizerPatchResult` | 补丁回调返回类型。 |

---

## 6. CUDA Runtime API（示例算子侧）

> 头文件 `cuda_runtime.h`；见 `examples/vec_add_oob.cu`。
> 用于构造一个"故意越界写"的样例算子来验证两种检测模式。**注意：这些属于用户程序侧的普通 CUDA 调用，不在 sanitizer 回调内。**

| API | 分类 | 在本项目中的作用 |
|-----|------|------------------|
| `cudaMalloc(&ptr, bytes)` | 内存分配 | 分配输入 `a`/`b` 与只够 `n` 个元素的输出 `c`（故意偏小以制造越界）。这些分配会被 sanitizer 的 RESOURCE 回调捕获进合法区间表。 |
| `cudaMemset(ptr, 0, bytes)` | 内存初始化 | 初始化输入数组。 |
| `cudaFree(ptr)` | 内存释放 | 释放三个数组；释放会被 RESOURCE 回调捕获，从合法区间表移除。 |
| `cudaDeviceSynchronize()` | 同步 | 等待越界 kernel `vecAddBad` 执行完成。 |
| kernel 启动 `vecAddBad<<<blocks, threads>>>(...)` | 内核启动 | 启动会触发 LAUNCH_BEGIN/END 回调，是检测的核心时机。 |

---

## 7. 外挂式注入入口（CUDA Driver 官方钩子）

> 见 `src/external_inject.cpp`。不属于函数式 API，而是驱动约定的**注入契约**。

| 符号 / 机制 | 作用 |
|-------------|------|
| `extern "C" int InitializeInjection(void)` | 当 `CUDA_INJECTION64_PATH` 指向本 `.so` 时，CUDA 驱动在初始化期间（首个 context 之前）`dlopen()` 并调用此入口。与 Compute Sanitizer 自身同款机制，保证 `sanitizerSubscribe()` 在任何用户 CUDA 调用前完成。它复用引擎 `oobEngineInitEx()`，并注册 `atexit` 在退出时打印汇总。 |
| `dladdr()` / `Dl_info`（`dlfcn.h`） | 解析本 `.so` 自身路径，从而在**旁边**寻找默认的 `oob_patch.cubin`（当未设 `OOB_PATCH_CUBIN` 时）。 |
| `readlink("/proc/self/exe", ...)`（嵌入式回退） | 引擎侧解析可执行文件路径，在其旁边寻找 `oob_patch.cubin`。 |

---

## 8. 补丁编译（构建期，非运行时 API）

| 命令 / 选项 | 作用 |
|-------------|------|
| `nvcc --cubin --compile-as-tools-patch oob_patch.cu -o oob_patch.cubin -arch=sm_80` | 把设备补丁编译成 sanitizer 可加载的 tools-patch cubin。`--compile-as-tools-patch` 是关键：产物不是普通 kernel，而是可被 `sanitizerAddPatchesFromFile` 拼接的补丁。 |

---

## 附：主机侧共享数据结构（host ↔ device 布局一致）

| 结构体 | 作用 |
|--------|------|
| `OobAllocEntry {base, size}` | 一条合法分配区间 `[base, base+size)`。 |
| `OobReport {pc, address, accessSize, flags, block/thread}` | 设备补丁写入的一条越界事件。 |
| `OobDeviceCtx {allocBase, numAllocs, reportBase, maxReports, reportCursor, abortOnError}` | 作为 `userdata` 传给补丁的控制块，host/device 布局必须完全一致。 |

---

## 附：API 数量小结

| 类别 | API 个数（近似） |
|------|------------------|
| Sanitizer Callback | 5 |
| Sanitizer Patching | 3 |
| Sanitizer Memory | 6 |
| Sanitizer Launch | 1 |
| Sanitizer 设备补丁 | 1（+2 返回值常量） |
| CUDA Runtime（示例侧） | 5 |

> 说明：本项目**不使用 CUPTI**，全部插桩/检测依赖 Compute Sanitizer API；
> 外挂注入入口、`dlfcn`/`readlink` 为路径解析辅助，不计入 Sanitizer API 统计。
