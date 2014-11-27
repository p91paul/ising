#include "stdio.h"
#include "common.h"
#include "random.h"

__device__   __forceinline__ dim3 getIndex() {
    dim3 index;
    index.x = blockIdx.x * BLOCK_SIZE_XY + threadIdx.x;
    index.y = blockIdx.y * BLOCK_SIZE_XY + threadIdx.y;
    index.z = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z;
    //printf("(%d,%d,%d) (%d,%d,%d)\n",blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
    return index;
}

template<int offsetX = L * L, int offsetY = L>
__device__ __forceinline__ unsigned int getTid(dim3 index) {
    return index.x * offsetX + index.y * offsetY + index.z;
}

static const int sharedXY = (BLOCK_SIZE_XY + 2);
static const int sharedZ = (BLOCK_SIZE_Z * 2 + 2);
static const int offsetX = sharedXY * sharedZ;
static const int offsetY = sharedZ;
static const int sharedSize = sharedXY * sharedXY * sharedZ;
static const int pos000 = sharedXY * sharedZ + sharedZ + 1;

template<char dir, int skip, int sizeXY = sharedXY, int sizeZ = sharedZ>
__device__ __forceinline__ int unsafeNeigh(int tid) {
    if (dir == 'x')
        return tid + skip * sizeXY * sizeZ;
    else if (dir == 'y')
        return tid + skip * sizeZ;
    else
        return tid + skip;
}

template<char dir, int skip, int size = L3, int sizeXY = L, int sizeZ = L>
__device__ __forceinline__ int neigh(int tid) {
    int neigh = unsafeNeigh<dir, skip, sizeXY, sizeZ>(tid);
    // this conditional statement is faster than (size+neigh) % size
    if (neigh < 0)
        return neigh + size;
    else if (neigh >= size)
        return neigh - size;
    return neigh;
}

template<bool second> __global__ void generateNext(int* S, float beta,
        curandState * const rngStates) {

    dim3 index = getIndex();
    //shifting to do inversions with the pattern of squares colours on a chess board
    int shifting = (index.x ^ index.y) & 1;
    if (second)
        shifting = 1 - shifting;

    index.z <<= 1;
    int tid = getTid(index);

    //double z coordinate
    int sTid = getTid<offsetX, offsetY>(threadIdx) + pos000 + threadIdx.z;

    __shared__ int sS[sharedSize];
    sS[sTid + (1 - shifting)] = S[tid + (1 - shifting)];
    sTid += shifting;
    tid += shifting;
    if (threadIdx.x == 0)
        sS[unsafeNeigh<'x', -1>(sTid)] = S[neigh<'x', -1>(tid)];
    if (threadIdx.y == 0)
        sS[unsafeNeigh<'y', -1>(sTid)] = S[neigh<'y', -1>(tid)];
    if (!shifting && threadIdx.z == 0)
        sS[unsafeNeigh<'z', -1>(sTid)] = S[neigh<'z', -1>(tid)];
    if (threadIdx.x == BLOCK_SIZE_XY - 1)
        sS[unsafeNeigh<'x', +1>(sTid)] = S[neigh<'x', +1>(tid)];
    if (threadIdx.y == BLOCK_SIZE_XY - 1)
        sS[unsafeNeigh<'y', +1>(sTid)] = S[neigh<'y', +1>(tid)];
    if (shifting && threadIdx.z == BLOCK_SIZE_Z - 1)
        sS[unsafeNeigh<'z', +1>(sTid)] = S[neigh<'z', +1>(tid)];
    __syncthreads();

    //printf("(%d,%d,%d)\n",index.x, index.y, index.z);
    //printf("%d\n", sTid + sharedXY * sharedZ);
    int nEnergy = sS[unsafeNeigh<'x', -1>(sTid)]
            + sS[unsafeNeigh<'x', +1>(sTid)] + sS[unsafeNeigh<'y', -1>(sTid)]
            + sS[unsafeNeigh<'y', +1>(sTid)] + sS[unsafeNeigh<'z', -1>(sTid)]
            + sS[unsafeNeigh<'z', +1>(sTid)];
    int cellS = S[tid];
    int dE = 2 * cellS * nEnergy;
    if (dE <= 0 || curand_uniform(&(rngStates[tid])) < expf(-beta * dE))
        S[tid] = -cellS;
}

template<bool second> __global__ void generateNextGlobal(int* S, float beta,
        curandState * const rngStates) {

    dim3 index = getIndex();
    //shifting to do inversions with the pattern of squares colours on a chess board
    int shifting = (index.x ^ index.y) & 1;
    if (second)
        shifting = 1 - shifting;

    index.z = (index.z << 1) | shifting;
    int tid = getTid(index);

    //printf("(%d,%d,%d)\n",index.x, index.y, index.z);
    //printf("%d\n", sTid + sharedXY * sharedZ);
    int nEnergy = S[neigh<'x', +1>(tid)] + S[neigh<'x', -1>(tid)]
            + S[neigh<'y', +1>(tid)] + S[neigh<'y', -1>(tid)]
            + S[neigh<'z', +1>(tid)] + S[neigh<'z', -1>(tid)];
    int cellS = S[tid];
    int dE = 2 * cellS * nEnergy;
    if (dE <= 0 || curand_uniform(&(rngStates[tid])) < __expf(-beta * dE))
        S[tid] = -cellS;
}

//compile all needed variants of template function
template __global__ void generateNext<true>(int* S, float beta,
        curandState * const rngStates);
template __global__ void generateNext<false>(int* S, float beta,
        curandState * const rngStates);
//global memory version
template __global__ void generateNextGlobal<true>(int* S, float beta,
        curandState * const rngStates);
template __global__ void generateNextGlobal<false>(int* S, float beta,
        curandState * const rngStates);

#include <stdio.h>

__global__ void print(int* S) {
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            for (int k = 0; k < L; ++k) {
                dim3 index(i, j, k);
                printf("%d,%d,%d: %d\n", i, j, k, S[getTid(index)]);
            }
}

__device__ __forceinline__ int energy(int* S, int tid) {
    int nEnergy = S[neigh<'x', 1>(tid)] + S[neigh<'x', -1>(tid)]
            + S[neigh<'y', 1>(tid)] + S[neigh<'y', -1>(tid)]
            + S[neigh<'z', 1>(tid)] + S[neigh<'z', -1>(tid)];
    return -S[tid] * nEnergy;

}

__global__ void totalEnergy(int* S) {
    int e = 0;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            for (int k = 0; k < L; ++k) {
                dim3 index(i, j, k);
                e += energy(S, getTid(index));
            }
    printf("Total energy= %d\n", e);
}
