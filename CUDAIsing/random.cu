#include <stdio.h>
#include "common.h"
#include "random.h"


__device__ __forceinline__ int randomSpin(curandState * const rngStates, unsigned int tid) {
    unsigned int rnd = curand(&rngStates[tid]);
    int binary = (rnd >> ((rnd ^ tid) & 31)) & 1;
    //printf("%d\n", (binary << 1) - 1);
    return (binary << 1) - 1;
}

__global__ void initRNG(curandState * const rngStates,
        const unsigned int seed) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < L3) {
        curand_init(seed, tid, 0, &rngStates[tid]);
    }
}

__global__ void fillMatrix(int* S, curandState * const rngStates) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < L3) {
        //skipahead(100, &rngStates[tid]);
        S[tid] = randomSpin(rngStates, tid);
    }
}
