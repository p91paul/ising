#include "common.h"
#include "random.h"

__device__ dim3 getIndex() {
    dim3 index;
    index.x = blockIdx.x * BLOCK_SIZE_XY + threadIdx.x;
    index.y = blockIdx.y * BLOCK_SIZE_XY + threadIdx.y;
    index.z = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z;
    //printf("(%d,%d,%d) (%d,%d,%d)\n",blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
    return index;
}

__device__ unsigned int getTid(dim3 index) {
    return index.x * L * L + index.y * L + index.z;
}

template<char dir, int skip> __device__ int neigh(dim3 index) {
    dim3 result = index;
    if (dir == 'x')
        result.x = (result.x + skip) % L;
    if (dir == 'y')
        result.y = (result.y + skip) % L;
    if (dir == 'z')
        result.z = (result.z + skip) % L;
    return getTid(result);
}

__device__ int energy(int* S, dim3 index, int tid) {
    int nEnergy = S[neigh<'x', 1>(index)] + S[neigh<'x', -1>(index)] + S[neigh<'y', 1>(index)]
            + S[neigh<'y', -1>(index)] + S[neigh<'z', 1>(index)] + S[neigh<'z', -1>(index)];
    return -S[tid] * nEnergy;
}

__device__ void tryInvert(int* S, dim3 index, float beta,
        curandState * const rngStates) {
    if (index.x < L && index.y < L && index.z < L) {
        unsigned int tid = getTid(index);
        int dE = -2 * energy(S, index, tid);
        if (dE < 0 || curand_uniform(&(rngStates[tid])) < __expf(-beta * dE))
            S[tid] = -S[tid];
    }
}

template<bool second> __global__ void generateNext(int* S, float beta, curandState * const rngStates) {
    dim3 index = getIndex();
    int shifting = (index.x ^ index.y) & 1;
    //printf("(%d,%d,%d) shifting %d second %d\n",index.x, index.y, index.z, shifting, second);
    if (second)
        shifting = 1 - shifting;
    index.z = (2 * index.z) + shifting;
    //printf("(%d,%d,%d)\n",index.x, index.y, index.z);
    tryInvert(S, index, beta, rngStates);
}

template __global__ void generateNext<true>(int* S, float beta, curandState * const rngStates);
template __global__ void generateNext<false>(int* S, float beta, curandState * const rngStates);


#include <stdio.h>

__global__ void print(int* S) {
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            for (int k = 0; k < L; ++k) {
                dim3 index(i, j, k);
                printf("%d,%d,%d: %d\n", i, j, k, S[getTid(index)]);
            }
}

__global__ void totalEnergy(int* S) {
    int e = 0;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            for (int k = 0; k < L; ++k) {
                dim3 index(i, j, k);
                e += energy(S, index, getTid(index));
            }
    printf("Total energy= %d\n", e);
}
