/*
 * external_inject.cpp - External-mode front-end (CUDA_INJECTION64_PATH).
 *
 * The CUDA driver dlopen()s the library named by the CUDA_INJECTION64_PATH
 * environment variable *before* the first CUDA context is created, then calls
 * the exported entry point:
 *
 *     extern "C" int InitializeInjection(void);
 *
 * This is the same official injection hook that Compute Sanitizer itself uses,
 * which guarantees our subscriber is installed before any user CUDA call.
 *
 * Configuration is taken from environment variables so the target binary needs
 * no source changes:
 *     OOB_PATCH_CUBIN     path to oob_patch.cubin (default: next to this .so)
 *     OOB_ABORT_ON_ERROR  1 -> abort on first OOB (default 0)
 *     OOB_VERBOSE         1 -> print each event (default 1)
 *     OOB_MAX_REPORTS     device report capacity (default 4096)
 */
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <dlfcn.h>

extern "C" int  oobEngineInitEx(const char* patchCubin,
                                int abortOnError, int verbose, int maxReports);
extern "C" void oobEngineShutdown(void);

static int envInt(const char* name, int def) {
    const char* v = getenv(name);
    if (!v || !*v) return def;
    return atoi(v);
}

/* Resolve the cubin path: explicit env, else next to this shared object. */
static std::string resolveCubin() {
    if (const char* e = getenv("OOB_PATCH_CUBIN")) return std::string(e);
    Dl_info info;
    if (dladdr((void*)&resolveCubin, &info) && info.dli_fname) {
        std::string p(info.dli_fname);
        size_t slash = p.find_last_of('/');
        if (slash != std::string::npos)
            return p.substr(0, slash + 1) + "oob_patch.cubin";
    }
    return std::string("oob_patch.cubin");
}

static void onExit() {
    oobEngineShutdown();
}

extern "C" int InitializeInjection(void) {
    /*
     * WHY this exact function name / mechanism:
     *   When CUDA_INJECTION64_PATH points at this .so, the CUDA driver dlopen()s
     *   it during driver initialization (BEFORE the first context is created)
     *   and calls InitializeInjection(). This is the officially supported
     *   injection hook (the same one Compute Sanitizer itself uses), which
     *   guarantees our sanitizerSubscribe() runs before any user CUDA call --
     *   satisfying the "subscribe first" requirement without touching the
     *   target program's source. Returning non-zero signals success.
     */
    static bool done = false;
    if (done) return 1;
    done = true;

    std::string cubin = resolveCubin();
    int abortOnError = envInt("OOB_ABORT_ON_ERROR", 0);
    int verbose      = envInt("OOB_VERBOSE", 1);
    int maxReports   = envInt("OOB_MAX_REPORTS", 4096);

    if (verbose) {
        fprintf(stderr, "[OOB] external injection active. patch=%s\n",
                cubin.c_str());
    }

    int rc = oobEngineInitEx(cubin.c_str(), abortOnError, verbose, maxReports);
    if (rc != 0) {
        fprintf(stderr, "[OOB] engine init failed (rc=%d); detection disabled\n", rc);
        return 0;
    }

    /*
     * Register a process-exit hook so the engine unsubscribes and prints its
     * summary even though the target program never calls oobStop() itself.
     */
    atexit(onExit);
    return 1; /* non-zero = success, per CUDA injection contract */
}
