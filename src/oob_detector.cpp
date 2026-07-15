/*
 * oob_detector.cpp - Embedded-mode front-end.
 * Implements the public C API declared in oob_detector.h by forwarding to
 * the shared engine (sanitizer_engine.cpp).
 */
#include "oob_detector.h"

#include <cstddef>

/* engine entry points (defined in sanitizer_engine.cpp) */
extern "C" int  oobEngineInitEx(const char* patchCubin,
                                int abortOnError, int verbose, int maxReports);
extern "C" void oobEngineShutdown(void);
extern "C" unsigned long long oobEngineErrorCount(void);
extern "C" void oobEngineSetAbort(int enable);

extern "C" int oobStart(const OobConfig* cfg) {
    OobConfig def = OOB_CONFIG_DEFAULT;
    if (!cfg) cfg = &def;
    return oobEngineInitEx(cfg->patchCubinPath,
                           cfg->abortOnError,
                           cfg->verbose,
                           cfg->maxReports > 0 ? cfg->maxReports : def.maxReports);
}

extern "C" void oobStop(void) {
    oobEngineShutdown();
}

extern "C" unsigned long long oobErrorCount(void) {
    return oobEngineErrorCount();
}

extern "C" void oobSetAbortOnError(int enable) {
    oobEngineSetAbort(enable);
}
