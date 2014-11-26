#include "common.h"
#include "random.h"


__device__ int randomSpin(curandState * const rngStates, unsigned int tid) {
    int rnd = curand(&rngStates[tid]);
    //printf("%f\n", rnd);
    int binary = (rnd >> ((rnd ^ tid) & 31)) & 1;
    return 2 * binary - 1;
}

__global__ void initRNG(curandState * const rngStates,
        const unsigned int seed) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < L3) {
        curand_init(seed, tid, 0, &rngStates[tid]);
    }
}

__global__ void fillMatrix(int* S, curandState * const rngStates) {
    unsigned int tid = blockIdx.x * SUM_BLOCK_SIZE + threadIdx.x;
    if (tid < L3) {
        //skipahead(100, &rngStates[tid]);
        S[tid] = randomSpin(rngStates, tid);
    }
}
