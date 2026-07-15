/*
 * oob_patch.cu - Compute Sanitizer device-side patch for global-memory
 * out-of-bounds detection.
 *
 * Compile with (see Makefile):
 *   nvcc --cubin --compile-as-tools-patch oob_patch.cu -o oob_patch.cubin -arch=sm_80
 *
 * The patch function `oob_global_access_cb` is registered for
 * SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS. It is invoked *before* every
 * global load/store/atomic. `userdata` points to an OobDeviceCtx placed in
 * device memory by the host engine (via sanitizerSetLaunchCallbackData).
 */

#include <sanitizer_patching.h>

/* Keep struct layout in sync with include/oob_detector.h.
 * We redefine here (device side) to avoid host-only includes. */

typedef struct OobAllocEntry {
    unsigned long long base;
    unsigned long long size;
} OobAllocEntry;

typedef struct OobReport {
    unsigned long long pc;
    unsigned long long address;
    unsigned int accessSize;
    unsigned int flags;
    unsigned int blockX, blockY, blockZ;
    unsigned int threadX, threadY, threadZ;
} OobReport;

typedef struct OobDeviceCtx {
    unsigned long long allocBase;   /* OobAllocEntry* */
    unsigned int numAllocs;
    unsigned int _pad0;
    unsigned long long reportBase;  /* OobReport* */
    unsigned int maxReports;
    unsigned int reportCursor;
    unsigned int abortOnError;
    unsigned int _pad1;
} OobDeviceCtx;

/* Access flags mirror Sanitizer_DeviceMemoryFlags */
#define OOB_FLAG_READ  0x1u
#define OOB_FLAG_WRITE 0x2u

/*
 * Returns 1 if [addr, addr+size) is fully inside some legal allocation.
 * Linear scan; allocation counts are typically small (< a few hundred).
 */
static __device__ __forceinline__
int oob_addr_is_legal(const OobDeviceCtx* ctx,
                      unsigned long long addr,
                      unsigned int size)
{
    const OobAllocEntry* allocs =
        (const OobAllocEntry*)ctx->allocBase;
    unsigned int n = ctx->numAllocs;
    unsigned long long end = addr + (unsigned long long)size;

    for (unsigned int i = 0; i < n; ++i) {
        unsigned long long b = allocs[i].base;
        unsigned long long e = b + allocs[i].size;
        /* fully contained */
        if (addr >= b && end <= e) {
            return 1;
        }
    }
    return 0;
}

extern "C" __device__ SanitizerPatchResult
oob_global_access_cb(void* userdata,
                     uint64_t pc,
                     void* ptr,
                     uint32_t accessSize,
                     uint32_t flags,
                     const void* /*pData*/)
{
    OobDeviceCtx* ctx = (OobDeviceCtx*)userdata;
    if (ctx == 0 || ctx->allocBase == 0) {
        /* engine not ready: don't interfere */
        return SANITIZER_PATCH_SUCCESS;
    }

    unsigned long long addr = (unsigned long long)ptr;

    /* A null / obviously-invalid pointer is also flagged. */
    if (oob_addr_is_legal(ctx, addr, accessSize)) {
        return SANITIZER_PATCH_SUCCESS;
    }

    /* Record the violation. Reserve a slot atomically. */
    unsigned int idx = atomicAdd(&ctx->reportCursor, 1u);
    if (idx < ctx->maxReports) {
        OobReport* reports = (OobReport*)ctx->reportBase;
        OobReport* r = &reports[idx];
        r->pc         = pc;
        r->address    = addr;
        r->accessSize = accessSize;
        /* keep only READ/WRITE bits we care about */
        r->flags      = flags & (OOB_FLAG_READ | OOB_FLAG_WRITE);
        r->blockX = blockIdx.x;  r->blockY = blockIdx.y;  r->blockZ = blockIdx.z;
        r->threadX = threadIdx.x; r->threadY = threadIdx.y; r->threadZ = threadIdx.z;
    }

    if (ctx->abortOnError) {
        /* Exits the whole warp; host will still read the report buffer. */
        return SANITIZER_PATCH_ERROR;
    }
    return SANITIZER_PATCH_SUCCESS;
}
