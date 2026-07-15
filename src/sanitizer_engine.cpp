/*
 * sanitizer_engine.cpp - Shared Compute Sanitizer OOB detection engine.
 *
 * Both front-ends (embedded oobStart/oobStop and the external injection .so)
 * call oobEngineInit()/oobEngineShutdown() from here.
 *
 * Responsibilities:
 *   - Subscribe to the Sanitizer callback API.
 *   - RESOURCE domain: track live device allocations (cudaMalloc/cudaFree),
 *     and patch each module when it is loaded.
 *   - LAUNCH domain: before each kernel launch, snapshot the allocation table
 *     into device memory and bind an OobDeviceCtx to the launch; after each
 *     launch, copy the report buffer back and print any OOB events.
 *
 * All device memory used for tracking is allocated with the Sanitizer Memory
 * API (sanitizerAlloc/...) because plain cudaMalloc must not be called from a
 * callback.
 */

#include "oob_detector.h"

#include <sanitizer.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <map>
#include <vector>
#include <string>
#include <atomic>

#include <unistd.h>   /* readlink */

namespace {

/* ----------------------------- config ----------------------------- */
struct EngineConfig {
    std::string patchCubin;
    int  abortOnError = 0;
    int  verbose      = 1;
    int  maxReports   = 4096;
};

/* --------------------------- global state -------------------------- */
struct PerLaunchBuffers {
    CUdeviceptr allocDev  = 0;   /* OobAllocEntry[]  (device) */
    CUdeviceptr reportDev = 0;   /* OobReport[]      (device) */
    CUdeviceptr ctxDev    = 0;   /* OobDeviceCtx     (device) */
    uint32_t    allocCap  = 0;   /* capacity of allocDev in entries */
};

struct Engine {
    Sanitizer_SubscriberHandle subscriber = nullptr;
    EngineConfig cfg;

    std::mutex mtx;

    /* live device allocations: base -> size */
    std::map<uint64_t, uint64_t> liveAllocs;

    /* one reusable set of device buffers per (context,stream) is overkill;
       we keep a small pool keyed by launch handle lifetime. For simplicity
       we allocate fresh buffers lazily and cache one set per context. */
    std::map<CUcontext, PerLaunchBuffers> bufs;

    std::atomic<unsigned long long> totalErrors{0};
    unsigned long long launchCount = 0;

    bool patchesLoadedForCtx = false;
    std::map<CUcontext, bool> ctxPatched;

    bool active = false;
};

Engine g;

/* --------------------------- helpers ------------------------------- */
#define SAN_CHECK(call)                                                    \
    do {                                                                   \
        SanitizerResult _r = (call);                                       \
        if (_r != SANITIZER_SUCCESS) {                                     \
            const char* _s = nullptr;                                      \
            sanitizerGetResultString(_r, &_s);                             \
            fprintf(stderr, "[OOB] sanitizer error %d (%s) at %s:%d\n",    \
                    (int)_r, _s ? _s : "?", __FILE__, __LINE__);           \
        }                                                                  \
    } while (0)

const char* resolvePatchPath(const EngineConfig& cfg) {
    static std::string cached;
    if (!cfg.patchCubin.empty()) { cached = cfg.patchCubin; return cached.c_str(); }
    if (const char* e = getenv("OOB_PATCH_CUBIN")) { cached = e; return cached.c_str(); }
    /* fall back: alongside the executable */
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = 0;
        std::string p(buf);
        size_t slash = p.find_last_of('/');
        cached = (slash == std::string::npos ? std::string() : p.substr(0, slash + 1))
                 + "oob_patch.cubin";
        return cached.c_str();
    }
    cached = "oob_patch.cubin";
    return cached.c_str();
}

/* Load patches into a context exactly once and mark it. Must be serialized. */
void ensurePatchesLoaded(CUcontext ctx) {
    if (g.ctxPatched.count(ctx) && g.ctxPatched[ctx]) return;
    const char* path = resolvePatchPath(g.cfg);
    /*
     * WHY sanitizerAddPatchesFromFile:
     *   The device patch (oob_patch.cu) is compiled separately into a cubin
     *   with `nvcc --compile-as-tools-patch`. It is NOT part of the user's
     *   program, so the sanitizer must load it into the *same CUDA context*
     *   as the user code before it can be spliced in. Patches are per-context
     *   (that is why we key ctxPatched by CUcontext and load once per ctx):
     *   loading twice is wasted work, and the Add/Patch family requires
     *   serialized access (hence g.mtx is held by the caller).
     */
    SanitizerResult r = sanitizerAddPatchesFromFile(path, ctx);
    if (r != SANITIZER_SUCCESS) {
        const char* s = nullptr; sanitizerGetResultString(r, &s);
        fprintf(stderr, "[OOB] failed to load patch cubin '%s': %d (%s)\n",
                path, (int)r, s ? s : "?");
    } else if (g.cfg.verbose) {
        fprintf(stderr, "[OOB] loaded device patch: %s\n", path);
    }
    g.ctxPatched[ctx] = (r == SANITIZER_SUCCESS);
}

/* Ensure device buffers exist for a context; (re)allocate alloc table if too small. */
PerLaunchBuffers& ensureBuffers(CUcontext ctx, uint32_t neededAllocs) {
    PerLaunchBuffers& b = g.bufs[ctx];

    if (b.ctxDev == 0) {
        void* p = nullptr;
        /*
         * WHY sanitizerAlloc (not cudaMalloc):
         *   This runs inside a sanitizer callback. Calling the CUDA runtime
         *   (cudaMalloc) from a callback is unsupported and can deadlock the
         *   application. sanitizerAlloc is the callback-safe equivalent and
         *   returns a normal device pointer we can hand to the device patch.
         *   This buffer holds the single OobDeviceCtx the patch reads.
         */
        SAN_CHECK(sanitizerAlloc(ctx, &p, sizeof(OobDeviceCtx)));
        b.ctxDev = (CUdeviceptr)p;
    }
    if (b.reportDev == 0) {
        void* p = nullptr;
        /* Callback-safe allocation of the report ring the device patch fills. */
        SAN_CHECK(sanitizerAlloc(ctx, &p,
                  sizeof(OobReport) * (size_t)g.cfg.maxReports));
        b.reportDev = (CUdeviceptr)p;
    }
    if (neededAllocs == 0) neededAllocs = 1;
    if (b.allocCap < neededAllocs) {
        /*
         * Grow the legal-interval table only when the current one is too
         * small (with headroom) to avoid re-allocating every launch. Free the
         * old buffer with sanitizerFree (callback-safe counterpart of cudaFree).
         */
        if (b.allocDev) { SAN_CHECK(sanitizerFree(ctx, (void*)b.allocDev)); b.allocDev = 0; }
        void* p = nullptr;
        uint32_t cap = neededAllocs + 64; /* headroom */
        SAN_CHECK(sanitizerAlloc(ctx, &p, sizeof(OobAllocEntry) * (size_t)cap));
        b.allocDev = (CUdeviceptr)p;
        b.allocCap = cap;
    }
    return b;
}

/* ----------------------- resource callbacks ------------------------ */
void onResource(Sanitizer_CallbackId cbid, const void* cbdata) {
    switch (cbid) {
        case SANITIZER_CBID_RESOURCE_MODULE_LOADED: {
            const auto* d = (const Sanitizer_ResourceModuleData*)cbdata;
            std::lock_guard<std::mutex> lk(g.mtx);
            ensurePatchesLoaded(d->context);
            /*
             * WHY patch at MODULE_LOADED:
             *   Instrumentation is per-CUmodule. A module contains the compiled
             *   kernels; we must inject our patch into it *after* it is loaded
             *   but *before* any of its kernels launch. This callback is the
             *   exact hook for that. Doing it lazily at launch time would race
             *   the first launch.
             *
             * sanitizerPatchInstructions(GLOBAL_MEMORY_ACCESS, module, name):
             *   Tells the sanitizer "for every global load/store/atomic in this
             *   module, call the device function `oob_global_access_cb`".
             *   We only pick GLOBAL access (not shared/local) because global
             *   OOB is the target of this detector; this keeps overhead down.
             *
             * sanitizerPatchModule(module):
             *   Actually rewrites the module's SASS to insert the selected
             *   callbacks. Must be called after PatchInstructions and before
             *   the kernels run.
             */
            SAN_CHECK(sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS,
                d->module, "oob_global_access_cb"));
            SAN_CHECK(sanitizerPatchModule(d->module));
            break;
        }
        case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC: {
            const auto* d = (const Sanitizer_ResourceMemoryData*)cbdata;
            std::lock_guard<std::mutex> lk(g.mtx);
            /*
             * WHY track allocations here:
             *   The device patch needs to know which address ranges are legal.
             *   The RESOURCE domain reports every device allocation (cudaMalloc,
             *   cudaMallocAsync, module-static memory, etc.) with its base and
             *   size. We record base->size so we can later hand the patch a
             *   snapshot of all live [base, base+size) intervals.
             */
            g.liveAllocs[d->address] = d->size;
            break;
        }
        case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_FREE: {
            const auto* d = (const Sanitizer_ResourceMemoryData*)cbdata;
            std::lock_guard<std::mutex> lk(g.mtx);
            /* Drop freed ranges so an access to freed memory is also flagged. */
            g.liveAllocs.erase(d->address);
            break;
        }
        default: break;
    }
}

/* --------------------------- reporting ----------------------------- */
void printReports(const char* kernelName,
                  const OobReport* reps, uint32_t count, uint32_t attempted) {
    for (uint32_t i = 0; i < count; ++i) {
        const OobReport& r = reps[i];
        const char* rw = (r.flags & OOB_FLAG_WRITE) ? "WRITE"
                        : (r.flags & OOB_FLAG_READ) ? "READ" : "ACCESS";

        /* find nearest allocation for a helpful "overflow by N" hint */
        uint64_t nb = 0, ns = 0; long long over = 0; bool have = false;
        {
            std::lock_guard<std::mutex> lk(g.mtx);
            for (auto& kv : g.liveAllocs) {
                uint64_t b = kv.first, e = b + kv.second;
                if (r.address >= b && r.address < e + kv.second) { /* near */
                    nb = b; ns = kv.second; have = true;
                    over = (long long)(r.address + r.accessSize) - (long long)e;
                    break;
                }
            }
        }

        fprintf(stderr,
            "========== [OOB] Out-of-bounds GLOBAL access detected ==========\n"
            "  kernel   : %s\n"
            "  access   : %s  size=%u bytes\n"
            "  address  : 0x%llx",
            kernelName ? kernelName : "(unknown)",
            rw, r.accessSize, (unsigned long long)r.address);
        if (have) {
            fprintf(stderr,
                "   (nearest alloc: base=0x%llx size=%llu",
                (unsigned long long)nb, (unsigned long long)ns);
            if (over > 0)
                fprintf(stderr, ", overflow by %lld bytes past end)", over);
            else
                fprintf(stderr, ")");
        }
        fprintf(stderr,
            "\n  thread   : block=(%u,%u,%u) thread=(%u,%u,%u)\n"
            "  pc       : 0x%llx\n"
            "================================================================\n",
            r.blockX, r.blockY, r.blockZ, r.threadX, r.threadY, r.threadZ,
            (unsigned long long)r.pc);
    }
    if (attempted > count) {
        fprintf(stderr, "[OOB] (%u more OOB access(es) not shown; increase "
                        "OOB_MAX_REPORTS)\n", attempted - count);
    }
}

/* ------------------------ launch callbacks ------------------------- */
void onLaunchBegin(const Sanitizer_LaunchData* d) {
    std::lock_guard<std::mutex> lk(g.mtx);

    uint32_t nAllocs = (uint32_t)g.liveAllocs.size();
    if (nAllocs > OOB_MAX_ALLOCS) nAllocs = OOB_MAX_ALLOCS;

    PerLaunchBuffers& b = ensureBuffers(d->context, nAllocs);

    /* build host alloc table snapshot */
    std::vector<OobAllocEntry> host(nAllocs);
    uint32_t i = 0;
    for (auto& kv : g.liveAllocs) {
        if (i >= nAllocs) break;
        host[i].base = kv.first;
        host[i].size = kv.second;
        ++i;
    }

    /* upload alloc table */
    if (nAllocs > 0) {
        Sanitizer_StreamHandle hs = d->hStream;
        /*
         * WHY sanitizerMemcpyHostToDeviceAsync (not cudaMemcpy):
         *   We are inside the LAUNCH_BEGIN callback. The callback-safe copy
         *   pushes the freshly snapshotted legal-interval table onto the SAME
         *   stream as the kernel, so it is guaranteed to land in device memory
         *   before the kernel (and thus the patch) reads it. Using cudaMemcpy
         *   here would be unsupported inside a callback.
         */
        SAN_CHECK(sanitizerMemcpyHostToDeviceAsync(
            (void*)b.allocDev, host.data(),
            sizeof(OobAllocEntry) * nAllocs, hs));
    }

    /* zero the report buffer & fill ctx */
    Sanitizer_StreamHandle hs = d->hStream;
    /*
     * WHY sanitizerMemset:
     *   Reset the report ring (and its atomic cursor lives in ctx, reset
     *   below) so counts from a previous launch do not leak into this one.
     *   Callback-safe equivalent of cudaMemset, ordered on the kernel stream.
     */
    SAN_CHECK(sanitizerMemset((void*)b.reportDev, 0,
              sizeof(OobReport) * (size_t)g.cfg.maxReports, hs));

    OobDeviceCtx hctx;
    memset(&hctx, 0, sizeof(hctx));
    hctx.allocBase    = (uint64_t)b.allocDev;
    hctx.numAllocs    = nAllocs;
    hctx.reportBase   = (uint64_t)b.reportDev;
    hctx.maxReports   = (uint32_t)g.cfg.maxReports;
    hctx.reportCursor = 0;
    hctx.abortOnError = (uint32_t)g.cfg.abortOnError;

    /* Push the control block the device patch will read as its `userdata`. */
    SAN_CHECK(sanitizerMemcpyHostToDeviceAsync(
        (void*)b.ctxDev, &hctx, sizeof(hctx), hs));

    /*
     * WHY sanitizerSetLaunchCallbackData:
     *   This binds our OobDeviceCtx device pointer to THIS specific launch so
     *   that when the patched kernel runs, every call to oob_global_access_cb
     *   receives that pointer as `userdata`. It is per-launch (as opposed to
     *   sanitizerSetCallbackData which is per-CUfunction) which is exactly
     *   what we want: each launch gets a freshly reset report buffer and the
     *   allocation snapshot valid at that moment.
     */
    SAN_CHECK(sanitizerSetLaunchCallbackData(
        d->hLaunch, d->function, d->hStream, (const void*)b.ctxDev));
}

void onLaunchEnd(const Sanitizer_LaunchData* d) {
    CUcontext ctx = d->context;
    PerLaunchBuffers b;
    {
        std::lock_guard<std::mutex> lk(g.mtx);
        auto it = g.bufs.find(ctx);
        if (it == g.bufs.end()) return;
        b = it->second;
        g.launchCount++;
    }

    /*
     * WHY sanitizerStreamSynchronize before reading back:
     *   The kernel (and its patch callbacks) write the report buffer
     *   asynchronously. We must wait for the launch stream to drain so the
     *   device writes are complete and visible before we copy them to host.
     *   This is the callback-safe form of cudaStreamSynchronize.
     */
    SAN_CHECK(sanitizerStreamSynchronize(d->hStream));

    /*
     * Read back the control block first to learn reportCursor: the patch used
     * atomicAdd on it, so it tells us how many OOB accesses were ATTEMPTED
     * (which may exceed maxReports if there were more than the buffer holds).
     */
    OobDeviceCtx hctx;
    memset(&hctx, 0, sizeof(hctx));
    SAN_CHECK(sanitizerMemcpyDeviceToHost(
        &hctx, (void*)b.ctxDev, sizeof(hctx), d->hStream));

    uint32_t attempted = hctx.reportCursor;
    if (attempted == 0) return;

    uint32_t count = attempted < hctx.maxReports ? attempted : hctx.maxReports;
    std::vector<OobReport> reps(count);
    /* Copy only the valid records back for host-side formatting/printing. */
    SAN_CHECK(sanitizerMemcpyDeviceToHost(
        reps.data(), (void*)b.reportDev,
        sizeof(OobReport) * count, d->hStream));

    g.totalErrors.fetch_add(attempted);

    if (g.cfg.verbose) {
        printReports(d->functionName, reps.data(), count, attempted);
    } else {
        fprintf(stderr, "[OOB] kernel '%s': %u out-of-bounds access(es)\n",
                d->functionName ? d->functionName : "(unknown)", attempted);
    }

    if (g.cfg.abortOnError) {
        fprintf(stderr, "[OOB] abortOnError set -> aborting.\n");
        fflush(stderr);
        abort();
    }
}

void onLaunch(Sanitizer_CallbackId cbid, const void* cbdata) {
    const auto* d = (const Sanitizer_LaunchData*)cbdata;
    if (cbid == SANITIZER_CBID_LAUNCH_BEGIN)      onLaunchBegin(d);
    else if (cbid == SANITIZER_CBID_LAUNCH_END)   onLaunchEnd(d);
}

/* ------------------------ master callback -------------------------- */
void SANITIZERAPI oobCallback(void* /*userdata*/,
                              Sanitizer_CallbackDomain domain,
                              Sanitizer_CallbackId cbid,
                              const void* cbdata) {
    switch (domain) {
        case SANITIZER_CB_DOMAIN_RESOURCE: onResource(cbid, cbdata); break;
        case SANITIZER_CB_DOMAIN_LAUNCH:   onLaunch(cbid, cbdata);   break;
        default: break;
    }
}

} /* anonymous namespace */

/* ------------------------- engine entry ---------------------------- */
/* C-friendly init that the front-ends fill in (fields passed explicitly). */
extern "C" int oobEngineInitEx(const char* patchCubin,
                               int abortOnError, int verbose, int maxReports) {
    if (g.active) return 0;
    g.cfg.patchCubin  = patchCubin ? patchCubin : "";
    g.cfg.abortOnError = abortOnError;
    g.cfg.verbose      = verbose;
    g.cfg.maxReports   = maxReports > 0 ? maxReports : 4096;

    SanitizerResult r = sanitizerSubscribe(&g.subscriber, oobCallback, nullptr);
    /*
     * WHY sanitizerSubscribe first, before any CUDA call:
     *   A subscriber must exist before the CUDA driver initializes / creates
     *   the first context, otherwise early allocations and module loads are
     *   missed and the allocation table / patches would be incomplete. Only
     *   ONE subscriber may exist at a time (docs); we keep the handle to
     *   unsubscribe cleanly in oobEngineShutdown.
     */
    if (r != SANITIZER_SUCCESS) {
        const char* s = nullptr; sanitizerGetResultString(r, &s);
        fprintf(stderr, "[OOB] sanitizerSubscribe failed: %d (%s)\n",
                (int)r, s ? s : "?");
        return (int)r;
    }
    /*
     * WHY enable only RESOURCE + LAUNCH domains (not all):
     *   RESOURCE gives us module-load (to patch) and alloc/free (to build the
     *   legal-interval table). LAUNCH gives us the per-kernel begin/end hooks
     *   to set callback data and harvest reports. Enabling other domains
     *   (driver/runtime API, memcpy, ...) would add callback overhead we do
     *   not need for OOB detection.
     */
    SAN_CHECK(sanitizerEnableDomain(1, g.subscriber, SANITIZER_CB_DOMAIN_RESOURCE));
    SAN_CHECK(sanitizerEnableDomain(1, g.subscriber, SANITIZER_CB_DOMAIN_LAUNCH));
    g.active = true;
    if (g.cfg.verbose)
        fprintf(stderr, "[OOB] engine initialized (abortOnError=%d, maxReports=%d)\n",
                g.cfg.abortOnError, g.cfg.maxReports);
    return 0;
}

extern "C" void oobEngineShutdown(void) {
    if (!g.active) return;
    if (g.subscriber) {
        /*
         * WHY sanitizerUnsubscribe:
         *   Releases the single global subscriber slot and stops all further
         *   callbacks. Required before the process exits (or before a new
         *   subscriber could be installed) to avoid dangling callbacks into
         *   freed engine state.
         */
        sanitizerUnsubscribe(g.subscriber);
        g.subscriber = nullptr;
    }
    unsigned long long errs = g.totalErrors.load();
    fprintf(stderr,
        "[OOB] Summary: %llu out-of-bounds access(es) across %llu kernel launch(es).\n",
        errs, g.launchCount);
    g.active = false;
}

extern "C" unsigned long long oobEngineErrorCount(void) {
    return g.totalErrors.load();
}

extern "C" void oobEngineSetAbort(int enable) {
    std::lock_guard<std::mutex> lk(g.mtx);
    g.cfg.abortOnError = enable;
}
