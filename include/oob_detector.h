/*
 * oob_detector.h - Public API + shared host/device data structures
 * for the CUDA device global-memory out-of-bounds (OOB) detector.
 *
 * Two front-ends share this file:
 *   - Embedded mode  (oobStart / oobStop), see oob_detector.cpp
 *   - External mode  (CUDA_INJECTION64_PATH .so), see external_inject.cpp
 *
 * The structures OobAllocEntry / OobReport / OobDeviceCtx are also included
 * by the device patch (device/oob_patch.cu) so that host and device agree
 * on the memory layout of the callback userdata buffer.
 */
#ifndef OOB_DETECTOR_H
#define OOB_DETECTOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Shared host <-> device layout                                       */
/* ------------------------------------------------------------------ */

/* Maximum number of live allocations tracked per kernel launch.       */
#ifndef OOB_MAX_ALLOCS
#define OOB_MAX_ALLOCS 4096
#endif

/* One legal device allocation interval [base, base+size).             */
typedef struct OobAllocEntry {
    uint64_t base;
    uint64_t size;
} OobAllocEntry;

/* Access-type flags mirrored from Sanitizer_DeviceMemoryFlags.        */
#define OOB_FLAG_READ  0x1u
#define OOB_FLAG_WRITE 0x2u

/* One recorded out-of-bounds event (written by the device patch).     */
typedef struct OobReport {
    uint64_t pc;        /* program counter of the offending instruction */
    uint64_t address;   /* accessed device address                      */
    uint32_t accessSize;/* bytes                                        */
    uint32_t flags;     /* OOB_FLAG_READ / OOB_FLAG_WRITE               */
    uint32_t blockX, blockY, blockZ;
    uint32_t threadX, threadY, threadZ;
} OobReport;

/*
 * Device-side context: this exact struct is passed as `userdata` to every
 * patch callback via sanitizerSetLaunchCallbackData(). It lives in device
 * memory allocated by the host engine (through sanitizerAlloc).
 *
 * Layout must stay identical between host (this header) and device patch.
 */
typedef struct OobDeviceCtx {
    /* legal allocation table (sorted or unsorted; linear scan on device) */
    uint64_t allocBase;      /* device pointer to OobAllocEntry[numAllocs] */
    uint32_t numAllocs;
    uint32_t _pad0;

    /* report ring: reports[0..maxReports-1], cursor is atomic counter    */
    uint64_t reportBase;     /* device pointer to OobReport[maxReports]    */
    uint32_t maxReports;
    uint32_t reportCursor;   /* atomicAdd cursor; total attempts (may > max) */

    uint32_t abortOnError;   /* if non-zero, patch returns PATCH_ERROR      */
    uint32_t _pad1;
} OobDeviceCtx;

/* ------------------------------------------------------------------ */
/* Embedded-mode public C API                                          */
/* ------------------------------------------------------------------ */

typedef struct OobConfig {
    const char* patchCubinPath; /* NULL -> $OOB_PATCH_CUBIN or exe dir  */
    int         abortOnError;   /* 1 -> abort process on first OOB      */
    int         verbose;        /* 1 -> print each OOB event            */
    int         maxReports;     /* device report buffer capacity        */
} OobConfig;

#define OOB_CONFIG_DEFAULT { NULL, 0, 1, 4096 }

/*
 * Initialize the detector and subscribe to Compute Sanitizer.
 * MUST be called before the first CUDA API call / context creation.
 * Returns 0 on success, non-zero on failure.
 */
int oobStart(const OobConfig* cfg);

/* Unsubscribe, flush pending reports, print summary, release resources. */
void oobStop(void);

/* Total number of OOB accesses detected so far. */
unsigned long long oobErrorCount(void);

/* Toggle abort-on-error at runtime. */
void oobSetAbortOnError(int enable);

#ifdef __cplusplus
}
#endif

#endif /* OOB_DETECTOR_H */
