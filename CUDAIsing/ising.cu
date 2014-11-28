#include "stdio.h"
#include "common.h"
#include "random.h"

__device__ __forceinline__ dim3 getIndex() {
    dim3 index;
    index.x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    index.y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    index.z = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z;
    //printf("(%d,%d,%d) (%d,%d,%d)\n",blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
    return index;
}

template<int offsetX = L * L, int offsetY = L>
__device__ __forceinline__ unsigned int getTid(dim3 index) {
    return index.x * offsetX + index.y * offsetY + index.z;
}

static const int sharedX = (BLOCK_SIZE_X + 2);
static const int sharedY = (BLOCK_SIZE_Y + 2);
static const int sharedZ = (BLOCK_SIZE_Z * 2 + 2);
static const int offsetX = sharedY * sharedZ;
static const int offsetY = sharedZ;
static const int pSharedX = BLOCK_SIZE_X;
static const int pSharedY = BLOCK_SIZE_Y;
static const int pSharedZ = BLOCK_SIZE_Z * 2;
static const int pOffsetX = pSharedY * pSharedZ;
static const int pOffsetY = pSharedZ;
static const int pSharedSize = pSharedX * pSharedY * pSharedZ;
static const int sharedSize = sharedX * sharedY * sharedZ;
static const int pos000 = sharedY * sharedZ + sharedZ + 1;

template<char dir, int skip, int sizeY = sharedY, int sizeZ = sharedZ>
__device__ __forceinline__ int unsafeNeigh(int tid) {
    if (dir == 'x')
        return tid + skip * sizeY * sizeZ;
    else if (dir == 'y')
        return tid + skip * sizeZ;
    else
        return tid + skip;
}

template<char dir, int skip, int size = L3, int sizeY = L, int sizeZ = L>
__device__ __forceinline__ int neigh(int tid) {
    int neigh = unsafeNeigh<dir, skip, sizeY, sizeZ>(tid);
    //return (size+neigh) % size;
    // this conditional statement is faster than (size+neigh) % size
    if (neigh < 0)
        return neigh + size;
    else if (neigh >= size)
        return neigh - size;
    return neigh;
}

template<bool second> __global__ void generateNextAllShared(int* S, float beta,
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
    int cellS = S[tid];
    if (threadIdx.x == 0)
        sS[unsafeNeigh<'x', -1>(sTid)] = S[neigh<'x', -1>(tid)];
    if (threadIdx.y == 0)
        sS[unsafeNeigh<'y', -1>(sTid)] = S[neigh<'y', -1>(tid)];
    if (!shifting && threadIdx.z == 0)
        sS[unsafeNeigh<'z', -1>(sTid)] = S[neigh<'z', -1>(tid)];
    if (threadIdx.x == BLOCK_SIZE_X - 1)
        sS[unsafeNeigh<'x', +1>(sTid)] = S[neigh<'x', +1>(tid)];
    if (threadIdx.y == BLOCK_SIZE_Y - 1)
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
    int dE = 2 * cellS * nEnergy;
    if (dE <= 0 || curand_uniform(&(rngStates[tid])) < __expf(-beta * dE))
        S[tid] = -cellS;
}

template<bool second> __global__ void generateNextPartlyShared(int* S,
        float beta, curandState * const rngStates) {

    dim3 index = getIndex();
    //shifting to do inversions with the pattern of squares colours on a chess board
    int shifting = (index.x ^ index.y) & 1;
    if (second)
        shifting = 1 - shifting;

    index.z <<= 1;
    int tid = getTid(index);

    //double z coordinate
    int sTid = getTid<pOffsetX, pOffsetY>(threadIdx) + threadIdx.z;

    __shared__ int sS[pSharedSize];
    sS[sTid + (1 - shifting)] = S[tid + (1 - shifting)];
    sTid += shifting;
    tid += shifting;
    __syncthreads();

    //printf("(%d,%d,%d)\n",index.x, index.y, index.z);
    //printf("%d\n", sTid + sharedXY * sharedZ);
    int cellS = S[tid];
    int left =
            threadIdx.x == 0 ?
                    S[neigh<'x', -1>(tid)] :
                    sS[unsafeNeigh<'x', -1, pSharedY, pSharedZ>(sTid)];
    int right =
            threadIdx.x == BLOCK_SIZE_X - 1 ?
                    S[neigh<'x', +1>(tid)] :
                    sS[unsafeNeigh<'x', +1, pSharedY, pSharedZ>(sTid)];
    int down =
            threadIdx.y == 0 ?
                    S[neigh<'y', -1>(tid)] :
                    sS[unsafeNeigh<'y', -1, pSharedY, pSharedZ>(sTid)];
    int up =
            threadIdx.y == BLOCK_SIZE_Y - 1 ?
                    S[neigh<'y', +1>(tid)] :
                    sS[unsafeNeigh<'y', +1, pSharedY, pSharedZ>(sTid)];
    int back =
            threadIdx.z == 0 && !shifting ?
                    S[neigh<'z', -1>(tid)] :
                    sS[unsafeNeigh<'z', -1, pSharedY, pSharedZ>(sTid)];
    int front =
            threadIdx.z == BLOCK_SIZE_Z - 1 && shifting ?
                    S[neigh<'z', +1>(tid)] :
                    sS[unsafeNeigh<'z', +1, pSharedY, pSharedZ>(sTid)];
    int nEnergy = left + right + down + up + back + front;
    int dE = 2 * cellS * nEnergy;
    if (dE <= 0 || curand_uniform(&(rngStates[tid])) < __expf(-beta * dE))
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
//all copied to shared memory version
template __global__ void generateNextAllShared<true>(int* S, float beta,
        curandState * const rngStates);
template __global__ void generateNextAllShared<false>(int* S, float beta,
        curandState * const rngStates);
//internal cells copied to shared memory version
template __global__ void generateNextPartlyShared<true>(int* S, float beta,
        curandState * const rngStates);
template __global__ void generateNextPartlyShared<false>(int* S, float beta,
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
