# Compute Sanitizer API 归类分析

> 数据来源：`/home/Cuda-13/compute-sanitizer`
> - 头文件：`include/*.h`
> - 文档：`docs/`（Doxygen 生成的 `api/`、`SanitizerApi/`、`SanitizerApiGuide/`）
> - 版本：`docs/VERSION` = **13.3.1**

本文对 **Compute Sanitizer Public API**（下称 Sanitizer API）做完整归类分析：按官方
`\defgroup` 把全部 **30 个 Host 函数** + 全部枚举 / 结构体 / 回调函数指针类型 分门别类，
并标注每一类的用途、关键类型、以及在本项目 `device-oob-detector` 中的使用情况。

---

## 0. 总览

### 0.1 头文件与分组对应关系

Sanitizer API 由 `sanitizer.h` 作为总入口，聚合 8 个子头文件。官方用 6 个 Doxygen 分组
（`\defgroup`）组织，与头文件、`docs/api/group__*.html` 一一对应：

| # | 分组 (`\defgroup`) | 头文件 | 文档页 | Host 函数数 |
|---|---|---|---|---|
| 1 | `SANITIZER_RESULT_API` | `sanitizer_result.h` | `group__SANITIZER__RESULT__API.html` | 1 |
| 2 | `SANITIZER_CALLBACK_API` | `sanitizer_callbacks.h` | `group__SANITIZER__CALLBACK__API.html` | 7 |
| 3 | `SANITIZER_PATCHING_API` | `sanitizer_patching.h` | `group__SANITIZER__PATCHING__API.html` | 11 |
| 4 | `SANITIZER_MEMORY_API` | `sanitizer_memory.h` | `group__SANITIZER__MEMORY__API.html` | 7 |
| 5 | `SANITIZER_STREAM_API` | `sanitizer_stream.h` | `group__SANITIZER__STREAM__API.html` | 3 |
| 6 | `SANITIZER_BARRIER_API` | `sanitizer_barrier.h` | `group__SANITIZER__BARRIER__API.html` | 1 |
| — | (回调 ID 常量表) | `sanitizer_driver_cbid.h` / `sanitizer_runtime_cbid.h` | — | 0 |
| — | (NVTX 指南) | — | `SanitizerNvtxGuide/` | 0 |
| | | | **合计** | **30** |

### 0.2 功能维度归类（跨分组的使用视角）

从"做一个 Sanitizer 工具需要哪几步"的角度，30 个函数可归纳为 5 大职能：

```
┌─────────────────────────────────────────────────────────────────────┐
│  A. 订阅与事件驱动 (Callback)   ── 骨架：订阅域、收 alloc/free/launch  │
│  B. 插桩与代码改写 (Patching)   ── 把 device 补丁 splice 进用户 kernel │
│  C. 回调内安全内存操作 (Memory) ── 回调里不能用 CUDA runtime，用它替代 │
│  D. 流与同步 (Stream)           ── 句柄互转 + 同步                     │
│  E. 元信息查询 (Result/Barrier) ── 错误串、barrier 数、PC/size 查询    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. `SANITIZER_RESULT_API` — 结果码与错误处理

> 头文件 `sanitizer_result.h`。所有 Sanitizer 函数的返回类型基础。

### Host 函数（1）

| 函数 | 作用 |
|---|---|
| `sanitizerGetResultString(SanitizerResult, const char** str)` | 把结果码转为可读字符串，线程安全。诊断/日志必备。 |

### 核心枚举 `SanitizerResult`

所有 API 的返回类型。关键取值：

| 值 | 含义 |
|---|---|
| `SANITIZER_SUCCESS` (0) | 成功 |
| `SANITIZER_ERROR_INVALID_PARAMETER` (1) | 参数非法 |
| `SANITIZER_ERROR_INVALID_DEVICE` (2) / `_INVALID_CONTEXT` (3) | 设备/上下文非法 |
| `SANITIZER_ERROR_INVALID_DOMAIN_ID` (4) / `_INVALID_CALLBACK_ID` (5) | 域/回调 ID 非法 |
| `SANITIZER_ERROR_INVALID_OPERATION` (6) | 依赖条件不满足 |
| `SANITIZER_ERROR_OUT_OF_MEMORY` (7) | 内存不足 |
| `SANITIZER_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT` (8) | 输出缓冲区不足 |
| `SANITIZER_ERROR_API_NOT_IMPLEMENTED` (9) / `_NOT_SUPPORTED` (14) | 未实现/不支持 |
| `SANITIZER_ERROR_MAX_LIMIT_REACHED` (10) | 达到上限 |
| `SANITIZER_ERROR_NOT_READY` (11) / `_NOT_COMPATIBLE` (12) | 未就绪/状态不兼容 |
| `SANITIZER_ERROR_NOT_INITIALIZED` (13) | 未初始化（连不上 driver） |
| `SANITIZER_ERROR_ADDRESS_NOT_IN_DEVICE_MEMORY` (15) | 地址不在 device 内存 |
| `SANITIZER_ERROR_UNKNOWN` (999) | 未知内部错误 |

> **本项目使用**：`sanitizerGetResultString` 用于 `SAN_CHECK` 宏诊断。

---

## 2. `SANITIZER_CALLBACK_API` — 回调订阅（工具骨架）

> 头文件 `sanitizer_callbacks.h`（最大，2200+ 行）。整个工具的事件驱动核心。

### Host 函数（7）

| 函数 | 分类 | 作用 |
|---|---|---|
| `sanitizerSubscribe(Sanitizer_SubscriberHandle*, Sanitizer_CallbackFunc, void* userdata)` | 订阅 | 注册**唯一**主回调；全局仅一个订阅者。必须早于首个 CUDA 调用。 |
| `sanitizerUnsubscribe(Sanitizer_SubscriberHandle)` | 退订 | 释放订阅者、停止后续回调。 |
| `sanitizerEnableDomain(uint32_t enable, subscriber, Sanitizer_CallbackDomain)` | 域开关 | 开/关某个回调域（RESOURCE / LAUNCH / …）。 |
| `sanitizerEnableAllDomains(uint32_t enable, subscriber)` | 域开关 | 一次性开/关所有域。 |
| `sanitizerEnableCallback(uint32_t enable, subscriber, domain, cbid)` | 单点开关 | 精细到单个 cbid 的开/关。 |
| `sanitizerGetCallbackState(uint32_t* enable, subscriber, domain, cbid)` | 查询 | 查询某回调当前是否启用。 |
| `sanitizerIsInsideRecursiveCallback(uint32_t* isInside)` | 查询 | 判断当前是否处于递归回调中（避免重入）。 |

### 回调函数类型

```c
typedef void (*Sanitizer_CallbackFunc)(void* userdata,
                                       Sanitizer_CallbackDomain domain,
                                       Sanitizer_CallbackId cbid,
                                       const void* cbdata);
```

### 回调域 `Sanitizer_CallbackDomain`（14 个）

| 域 | 值 | 用途 | 载荷结构 / cbid 枚举 |
|---|---|---|---|
| `_DRIVER_API` | 1 | CUDA Driver API 进出 | `Sanitizer_CallbackData` + `sanitizer_driver_cbid.h` |
| `_RUNTIME_API` | 2 | CUDA Runtime API 进出 | `Sanitizer_CallbackData` + `sanitizer_runtime_cbid.h` |
| `_RESOURCE` | 3 | 资源生命周期（**alloc/free/module**） | `Sanitizer_CallbackIdResource` + 多种 ResourceData |
| `_SYNCHRONIZE` | 4 | 同步事件 | `Sanitizer_CallackIdSync` / `Sanitizer_SynchronizeData` |
| `_LAUNCH` | 5 | kernel 启动 begin/end | `Sanitizer_CallbackIdLaunch` / `Sanitizer_LaunchData` |
| `_MEMCPY` | 6 | 内存拷贝 | `Sanitizer_CallbackIdMemcpy` / `Sanitizer_MemcpyData` |
| `_MEMSET` | 7 | 内存置位 | `Sanitizer_CallbackIdMemset` / `Sanitizer_MemsetData` |
| `_BATCH_MEMOP` | 8 | 批量内存操作 | `Sanitizer_BatchMemopData` |
| `_UVM` | 9 | 统一虚拟内存 | `Sanitizer_UvmData` |
| `_GRAPHS` | 10 | CUDA Graphs | `Sanitizer_GraphExec/NodeLaunch/LaunchData` |
| `_EVENTS` | 11 | CUDA events | `Sanitizer_EventData` |
| `_EXTERNAL_MEMORY` | 12 | 外部内存互操作 | `Sanitizer_ExternalMemoryData` |
| `_ERROR_LOGGING` | 13 | 错误日志 | `Sanitizer_ErrorLoggingData` |

### 关键 cbid（回调 ID）枚举

**`Sanitizer_CallbackIdResource`**（本项目重点用到）：
- `SANITIZER_CBID_RESOURCE_MODULE_LOADED` / `_MODULE_UNLOAD_STARTING` — 模块加载/卸载（打补丁时机）
- `SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC` / `_FREE` — device 内存分配/释放（维护区间表）
- 还含 context / stream / array / mempool / 虚拟地址范围 等资源事件

**`Sanitizer_CallbackIdLaunch`**：
- `SANITIZER_CBID_LAUNCH_BEGIN` / `_END` / `_AFTER_SYSCALL_SETUP` 等

**`Sanitizer_ApiCallbackSite`**：`SANITIZER_API_ENTER` / `SANITIZER_API_EXIT`（API 域进出点）。

### 回调载荷结构体（部分节选，共 20+ 个）

| 结构体 | 关键字段 | 用途 |
|---|---|---|
| `Sanitizer_CallbackData` | `functionName`、`functionParams`、`functionReturnValue` | Driver/Runtime API 载荷 |
| `Sanitizer_ResourceContextData` | `context`、`device` | context 创建/销毁 |
| `Sanitizer_ResourceStreamData` | `context`、`stream`、`hStream` | stream 创建/销毁 |
| `Sanitizer_ResourceModuleData` | `context`、`module`、`pCubin`、`cubinSize` | **模块加载 → 打补丁** |
| `Sanitizer_ResourceMemoryData` | `address`、`size`、`flags`、`permissions`、`visibility` | **alloc/free → 区间表** |
| `Sanitizer_ResourceMempoolData` / `ResourceArrayData` | pool / array 句柄 | 内存池 / CUDA array |
| `Sanitizer_LaunchData` | `context`、`module`、`function`、`hLaunch`、`hStream`、`functionName` | **launch begin/end** |
| `Sanitizer_MemcpyData` / `MemsetData` | 方向、大小、地址 | 拷贝/置位监控 |
| `Sanitizer_GraphExecData` / `GraphNodeLaunchData` / `GraphLaunchData` | graph/node 句柄 | Graphs 监控 |
| `Sanitizer_UvmData` / `EventData` / `ExternalMemoryData` / `ErrorLoggingData` | 各自事件字段 | UVM / 事件 / 外部内存 / 错误日志 |
| `Sanitizer_SynchronizeData` | `context`、`stream` | 同步事件 |

### 相关辅助枚举

- `Sanitizer_ResourceMemoryFlags` / `Sanitizer_ResourceMemoryPermissions` / `Sanitizer_MemoryVisibility` — 内存属性
- `Sanitizer_MemcpyDirection` — 拷贝方向
- `Sanitizer_BatchMemopType` / `Sanitizer_BatchMemopAtomicOp` — 批量内存操作类型

> **本项目使用**：订阅 `sanitizerSubscribe` + 开 `RESOURCE`/`LAUNCH` 两域
> （`sanitizerEnableDomain`），退出时 `sanitizerUnsubscribe`。核心结构：
> `Sanitizer_ResourceModuleData`（打补丁）、`Sanitizer_ResourceMemoryData`（区间表）、
> `Sanitizer_LaunchData`（launch 前后处理）。

---

## 3. `SANITIZER_PATCHING_API` — 指令级插桩（核心机制）

> 头文件 `sanitizer_patching.h`（1300+ 行）。把 device 补丁 splice 进用户 kernel 的机制。

### Host 函数（11）

| 函数 | 分类 | 作用 |
|---|---|---|
| `sanitizerAddPatchesFromFile(const char* filename, CUcontext)` | 加载补丁 | 从 cubin 文件加载补丁到 context（per-context）。 |
| `sanitizerAddPatches(const void* image, size_t, CUcontext)` | 加载补丁 | 同上，从内存镜像加载。 |
| `sanitizerPatchInstructions(Sanitizer_InstructionId, CUmodule, const char* cbName)` | 选插桩点 | 声明"某类指令都调用补丁函数 cbName"。 |
| `sanitizerPatchModule(CUmodule)` | 改写 | 真正把回调 splice 进 module 的 SASS。 |
| `sanitizerUnpatchModule(CUmodule)` | 还原 | 撤销 module 上的插桩。 |
| `sanitizerSetCallbackData(CUfunction, const void* userdata)` | 绑定数据 | 给某 kernel 绑定补丁 userdata。 |
| `sanitizerSetLaunchCallbackData(Sanitizer_LaunchHandle, CUfunction, Sanitizer_StreamHandle, void* userdata)` | 绑定数据 | 给**本次 launch**绑定补丁 userdata（推荐）。 |
| `sanitizerSetDeviceGraphData(CUgraphExec, ...)` | 绑定数据 | 给 device graph 绑定补丁数据。 |
| `sanitizerGetFunctionPcAndSize(CUmodule, CUfunction, uint64_t* pc, uint64_t* size)` | 查询 | 查函数 PC 与大小。 |
| `sanitizerGetCallbackPcAndSize(CUcontext, const char* cbName, uint64_t* pc, uint64_t* size)` | 查询 | 查补丁回调 PC 与大小。 |
| `sanitizerGetFunctionLoadedStatus(CUfunction, Sanitizer_FunctionLoadedStatus*)` | 查询 | 查函数加载状态（懒加载）。 |

### 核心枚举

**`Sanitizer_InstructionId`（45 种插桩点）** — 决定"插在哪类指令上"：

| 类别 | 取值 |
|---|---|
| 控制流 | `BLOCK_ENTER`(1), `BLOCK_EXIT`(2), `CALL`(9), `RET`(10) |
| **内存访问** | **`GLOBAL_MEMORY_ACCESS`(3)**, `SHARED_MEMORY_ACCESS`(4), `LOCAL_MEMORY_ACCESS`(5), `REMOTE_SHARED_MEMORY_ACCESS`(17), `MATRIX_MEMORY_ACCESS`(19) |
| 同步/barrier | `BARRIER`(6), `SYNCWARP`(7), `CUDA_BARRIER`(13), `CLUSTER_BARRIER_ARRIVE/WAIT`(21/22), `BARRIER_RELEASE`(29) |
| shuffle | `SHFL`(8) |
| device 端分配 | `DEVICE_SIDE_MALLOC`(11), `DEVICE_SIDE_FREE`(12), `DEVICE_ALIGNED_MALLOC`(18) |
| 异步拷贝/pipeline | `MEMCPY_ASYNC`(14), `PIPELINE_COMMIT`(15), `PIPELINE_WAIT`(16), `MEMCPY_ASYNC_BARRIER`(34) |
| cache | `CACHE_CONTROL`(20) |
| warpgroup/MMA | `WARPGROUP_MMA_ASYNC`(23), `WARPGROUP_WAIT_GROUP`(24), `WARPGROUP_FENCE`(25) |
| async store/reduce | `ASYNC_STORE`(26), `ASYNC_REDUCTION`(27) |
| 批量拷贝 (bulk/TMA) | `BULK_COPY_GLOBAL_TO_SHARED`(30), `BULK_COPY_SHARED_TO_GLOBAL`(35), `BULK_COPY_SHARED_TO_SHARED`(36), `BULK_REDUCTION_*`(37/38), `TMA_LOAD`(43), `TMA_STORE`(44), `TENSOR_CORE_BARRIER`(31) |
| 其它 | `MEMSET_SHARED`(32), `SET_SHARED_MEMORY_SIZE`(28) 等 |

**`SanitizerPatchResult`** — 补丁回调返回值：`SANITIZER_PATCH_SUCCESS` / `SANITIZER_PATCH_ERROR`（后者可终止 warp）。

**`Sanitizer_FunctionLoadedStatus`** / `Sanitizer_LoadMode` — 函数加载状态/模式。

其它枚举：`Sanitizer_DeviceMemoryFlags`、`Sanitizer_BarrierFlags`、`Sanitizer_CallFlags`、
`Sanitizer_CudaBarrierInstructionKind`、`Sanitizer_CacheControlInstructionKind`、
`Sanitizer_WarpgroupMMAAsyncFlags`。

### Device 端补丁回调函数指针类型（一个 InstructionId 对应一种签名）

补丁 `.cu` 里实现的函数需匹配对应签名。常用：

| 类型 | 对应指令 | 签名要点 |
|---|---|---|
| **`SanitizerCallbackMemoryAccess`** | GLOBAL/SHARED/LOCAL access | `(userdata, pc, void* ptr, accessSize, flags, pData)` |
| `SanitizerCallbackBlockEnter` / `BlockExit` | block enter/exit | `(userdata, pc)` |
| `SanitizerCallbackBarrier` | barrier | `(userdata, pc, barIndex, threadCount, flags)` |
| `SanitizerCallbackSyncwarp` / `Shfl` | syncwarp / shfl | `(userdata, pc, mask)` / `(userdata, pc)` |
| `SanitizerCallbackCall` / `Ret` | call / ret | `(userdata, pc, [targetPc, flags])` |
| `SanitizerCallbackDeviceSideMalloc` / `Free` | device 端 malloc/free | `(userdata, pc, ptr[, size])` |
| `SanitizerCallbackMemcpyAsync` / `MatrixMemoryAccess` / `CacheControl` | 异步拷贝/矩阵/cache | 各自专用参数 |
| `SanitizerCallbackCudaBarrier[Attempt]` / `Cluster*` / `Warpgroup*` / `Bulk*` / `Tensor*` / `Async*` | 新硬件特性 | Hopper/Blackwell 等专用 |

> **本项目使用**：`sanitizerAddPatchesFromFile` 加载 `oob_patch.cubin` →
> `sanitizerPatchInstructions(GLOBAL_MEMORY_ACCESS, module, "oob_global_access_cb")` →
> `sanitizerPatchModule` 改写 SASS；补丁函数按 `SanitizerCallbackMemoryAccess` 签名实现；
> launch 前用 `sanitizerSetLaunchCallbackData` 绑定 `OobDeviceCtx`。

---

## 4. `SANITIZER_MEMORY_API` — 回调内安全内存操作

> 头文件 `sanitizer_memory.h`。回调里**不能**用 CUDA runtime（`cudaMalloc` 等可能死锁），
> 用这套等价 API 替代。

### Host 函数（7）

| 函数 | 等价 CUDA | 作用 |
|---|---|---|
| `sanitizerAlloc(CUcontext, void** devPtr, size_t)` | `cudaMalloc` | 回调内 device 分配 |
| `sanitizerAllocHost(CUcontext, void** ptr, size_t)` | `cudaMallocHost` | 回调内 host pinned 分配 |
| `sanitizerFree(CUcontext, void* devPtr)` | `cudaFree` | 释放 device 内存 |
| `sanitizerFreeHost(CUcontext, void* ptr)` | `cudaFreeHost` | 释放 host 内存 |
| `sanitizerMemcpyHostToDeviceAsync(dst, src, count, Sanitizer_StreamHandle)` | `cudaMemcpyAsync` H2D | 上传（异步） |
| `sanitizerMemcpyDeviceToHost(dst, src, count, Sanitizer_StreamHandle)` | `cudaMemcpy` D2H | 回拷（完成即返回） |
| `sanitizerMemset(devPtr, value, count, Sanitizer_StreamHandle)` | `cudaMemset` | 置位 device 内存 |

> **本项目使用**：全部用到。`sanitizerAlloc/Free` 管理 device 端 ctx/report/alloc buffer；
> `MemcpyHostToDeviceAsync` 上传区间表+ctx；`MemcpyDeviceToHost` 回拷报告；`Memset` 清零。

---

## 5. `SANITIZER_STREAM_API` — 流与句柄互转

> 头文件 `sanitizer_stream.h`。Sanitizer 用 `Sanitizer_StreamHandle` 而非 `CUstream`。

### Host 函数（3）

| 函数 | 作用 |
|---|---|
| `sanitizerStreamSynchronize(Sanitizer_StreamHandle)` | 等价 `cudaStreamSynchronize`，回调内可用 |
| `sanitizerGetStream(Sanitizer_StreamHandle, CUstream* out)` | Sanitizer 句柄 → CUstream |
| `sanitizerGetStreamHandle(CUcontext, CUstream, Sanitizer_StreamHandle* out)` | CUstream → Sanitizer 句柄 |

### 核心类型
`typedef struct Sanitizer_Stream_st* Sanitizer_StreamHandle;`

> **本项目使用**：`sanitizerStreamSynchronize` 在 `onLaunchEnd` 回拷报告前等 device 写完。

---

## 6. `SANITIZER_BARRIER_API` — CUDA barrier 计数

> 头文件 `sanitizer_barrier.h`。

### Host 函数（1）

| 函数 | 作用 |
|---|---|
| `sanitizerGetCudaBarrierCount(CUfunction, uint32_t* numBarriers)` | 查询 kernel 使用的 CUDA barrier 数（需先 `sanitizerPatchModule`；仅 nvcc 11.2+ 编译的模块，否则返回 0）。 |

> **本项目使用**：未使用（OOB 检测不涉及 barrier 计数）。

---

## 7. 回调 ID 常量表（Driver / Runtime cbid）

> `sanitizer_driver_cbid.h`、`sanitizer_runtime_cbid.h`。无函数，仅两张巨大的枚举表，
> 为 `SANITIZER_CB_DOMAIN_DRIVER_API` / `_RUNTIME_API` 域列出每个 CUDA API 的 cbid，
> 例如 `SANITIZER_CBID_DRIVER_API_cuMemAlloc`、`SANITIZER_CBID_RUNTIME_API_cudaMalloc`。

配套 `include/generated_*_meta.h`（cuda / cuda_runtime_api / cudaGL / cudaVDPAU 等）
是各 CUDA API 的参数结构元信息，供解析 `Sanitizer_CallbackData.functionParams` 使用。

> **本项目使用**：未订阅 Driver/Runtime API 域（改用更高层的 RESOURCE 域捕获 alloc/free），
> 故未用到这两张表。

---

## 8. 文档资源（`docs/`）对照

| 目录/文件 | 内容 |
|---|---|
| `docs/api/group__*.html` | 上述 6 个分组的 Doxygen API 参考 |
| `docs/api/struct*.html` | 每个回调载荷结构体的字段说明 |
| `docs/api/modules.html` / `structs.html` | 分组/结构体总索引 |
| `docs/SanitizerApiGuide/index.html` | Sanitizer API 使用指南（教程） |
| `docs/SanitizerNvtxGuide/index.html` | NVTX 集成指南 |
| `docs/ComputeSanitizer/` | Compute Sanitizer 工具用户手册 |
| `docs/ReleaseNotes/` | 版本发布说明 |
| `docs/genindex.html` / `searchindex.js` | 全局索引/搜索 |

---

## 9. 归类结论

1. **按官方分组**：6 个 `\defgroup`，共 **30 个 Host 函数**；Callback(7) 与 Patching(11)
   两组占 60%，是构建任何 Sanitizer 工具的两大支柱。
2. **按职能**：订阅(A)→插桩(B)→回调内内存(C)→流同步(D)→查询(E) 构成一个工具的完整生命周期。
3. **可扩展面**：`Sanitizer_InstructionId` 有 45 种插桩点、`Sanitizer_CallbackDomain` 有 14 个域，
   本项目仅用了其中最小子集（GLOBAL 访存 + RESOURCE/LAUNCH 域）即实现了全局内存越界检测；
   若要扩展到 shared/local 越界、race、UAF、UVM 等，只需增开对应指令/域即可。

> **本项目（device-oob-detector）实际用到的 API 清单**详见 `CUDA_API.md` 与 `impl_details.md`
> 第十节；本文档提供的是 **Sanitizer API 全集的归类地图**，用于定位与扩展。
