/*
 * vec_add_oob.cu - Sample operator that deliberately writes out of bounds,
 * used to validate both detection modes.
 *
 * Embedded mode:
 *   nvcc vec_add_oob.cu -o vec_add_oob_embedded \
 *        -DOOB_EMBEDDED -I ../include -L ../build -loob_detector \
 *        -I $SANITIZER_DIR/include -L $SANITIZER_DIR -lsanitizer-public
 *
 * External mode (no special build):
 *   nvcc vec_add_oob.cu -o vec_add_oob
 *   ../tools/oob-check ./vec_add_oob
 */
#include <cstdio>
#include <cuda_runtime.h>

#ifdef OOB_EMBEDDED
#include "oob_detector.h"
#endif

/* Writes n+extra elements: the last `extra` writes overrun c[]. */
__global__ void vecAddBad(const float* a, const float* b, float* c,
                          int n, int extra) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n + extra) {           /* BUG: should be i < n */
        c[i] = a[i] + b[i];        /* out-of-bounds when i >= n */
    }
}

int main() {
#ifdef OOB_EMBEDDED
    /*
     * WHY oobStart before any CUDA call:
     *   It calls sanitizerSubscribe internally. The subscriber must be in
     *   place before the driver creates the first context, otherwise the
     *   early cudaMalloc/module-load events are missed and the detector
     *   cannot build its legal-address table or patch the kernels.
     */
    OobConfig cfg = OOB_CONFIG_DEFAULT;
    cfg.abortOnError = 0;
    cfg.verbose = 1;
    oobStart(&cfg);   /* must precede any CUDA call */
#endif

    const int n = 256;
    const int extra = 8;          /* intentional overflow */
    size_t bytes = n * sizeof(float);

    float *a, *b, *c;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&c, bytes);        /* only n elements! */
    cudaMemset(a, 0, bytes);
    cudaMemset(b, 0, bytes);

    int threads = 64;
    int blocks = (n + extra + threads - 1) / threads;
    vecAddBad<<<blocks, threads>>>(a, b, c, n, extra);
    cudaDeviceSynchronize();

    cudaFree(a); cudaFree(b); cudaFree(c);

#ifdef OOB_EMBEDDED
    printf("detected %llu OOB accesses\n", oobErrorCount());
    oobStop();
#endif
    printf("done\n");
    return 0;
}
