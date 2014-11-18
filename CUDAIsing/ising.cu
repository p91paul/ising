#include <curand_kernel.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <ctime>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace boost::random;

static const int SEED = 5;

static const int L = 32;
static const int L3 = L * L * L;
static const int BLOCKS_X = 4;
static const int BLOCK_SIZE = L / BLOCKS_X;

static const int SUM_NUM_BLOCKS = 32;
static const int SUM_BLOCK_SIZE = L3 / SUM_NUM_BLOCKS / 2;
//static const double B = 0;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		cerr << "Error " << _m_cudaStat << ": "                             \
                << cudaGetErrorString(_m_cudaStat) << " at line "	        \
				<< __LINE__ << " in file " << __FILE__ << endl;			    \
		exit(1);															\
	}																		\
}

__device__ dim3 getIndex() {
    dim3 index;
    index.x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    index.y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    index.z = blockIdx.z * BLOCK_SIZE + threadIdx.z;
    return index;
}

__device__ unsigned int getTid(dim3 index) {
    return index.x * L * L + index.y * L + index.z;

}

__device__ int randomSpin(curandState * const rngStates, unsigned int tid) {
    int rnd = curand(&rngStates[tid]);
    //printf("%f\n", rnd);
    int binary = rnd & 1;
    return 2 * binary - 1;
}

__global__ void initRNG(curandState * const rngStates,
        const unsigned int seed) {
    unsigned int tid = blockIdx.x * SUM_BLOCK_SIZE + threadIdx.x;
    if (tid < L3) {
        curand_init(seed, tid, 0, &rngStates[tid]);
    }
}

__global__ void fillMatrix(int* S, int* Jx, int* Jy, int* Jz,
        curandState * const rngStates) {
    unsigned int tid = blockIdx.x * SUM_BLOCK_SIZE + threadIdx.x;
    if (tid < L3) {
        //skipahead(100, &rngStates[tid]);
        S[tid] = randomSpin(rngStates, tid);
        Jx[tid] = randomSpin(rngStates, tid);
        Jy[tid] = randomSpin(rngStates, tid);
        Jz[tid] = randomSpin(rngStates, tid);
    }
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

__device__ int energy(int* S, int* Jx, int* Jy, int* Jz, dim3 index, int tid) {
    int prevX = neigh<'x', -1>(index);
    int prevY = neigh<'y', -1>(index);
    int prevZ = neigh<'z', -1>(index);
    int nEnergy = S[neigh<'x', 1>(index)] * Jx[tid] + S[prevX] * Jx[prevX]
            + S[neigh<'y', 1>(index)] * Jy[tid] + S[prevY] * Jy[prevY]
            + S[neigh<'z', 1>(index)] * Jz[tid] + S[prevZ] * Jz[prevZ];
    return -S[tid] * nEnergy;
}

__device__ void tryInvert(int* S, int* Jx, int* Jy, int* Jz, dim3 index,
        float beta, curandState * const rngStates) {
    if (index.x < L && index.y < L && index.z < L) {
        unsigned int tid = getTid(index);
        int dE = 2 * energy(S, Jx, Jy, Jz, index, tid);
        if (dE < 0 || curand_uniform(&(rngStates[tid])) < __expf(-beta * dE))
            S[tid] = -S[tid];
    }
}

__global__ void generateNext(int* S, int* Jx, int* Jy, int* Jz, float beta,
        curandState * const rngStates, int offset) {
    dim3 index = getIndex();
    index.z = offset + 2 * index.z + (index.x & 1) ^ (index.y & 1);
    tryInvert(S, Jx, Jy, Jz, index, beta, rngStates);
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory {
    __device__ inline operator T *() {
        extern __shared__ int __smem[];
        return (T *) __smem;
    }

    __device__ inline operator const T *() const {
        extern __shared__ int __smem[];
        return (T *) __smem;
    }
};

/*
 This version adds multiple elements per thread sequentially.  This reduces the overall
 cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
 (Brent's Theorem optimization)

 Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
 In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
 If blockSize > 32, allocate blockSize*sizeof(T) bytes.
 */
template<class T, unsigned int blockSize, bool nIsPow2>
__global__ void sum(T *g_idata, T *g_odata, unsigned int n) {
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n) {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i + blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256)) {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    __syncthreads();

    if ((blockSize >= 128) && (tid < 64)) {
        sdata[tid] = mySum = mySum + sdata[tid + 64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >= 64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >= 64) && (tid < 32)) {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >= 32) && (tid < 16)) {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >= 16) && (tid < 8)) {
        sdata[tid] = mySum = mySum + sdata[tid + 8];
    }

    __syncthreads();

    if ((blockSize >= 8) && (tid < 4)) {
        sdata[tid] = mySum = mySum + sdata[tid + 4];
    }

    __syncthreads();

    if ((blockSize >= 4) && (tid < 2)) {
        sdata[tid] = mySum = mySum + sdata[tid + 2];
    }

    __syncthreads();

    if ((blockSize >= 2) && (tid < 1)) {
        sdata[tid] = mySum = mySum + sdata[tid + 1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = mySum;
}

__global__ void print(int* S) {
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            for (int k = 0; k < L; ++k) {
                dim3 index(i, j, k);
                printf("%d,%d,%d: %d\n", i, j, k, S[getTid(index)]);
            }
}

__global__ void totalEnergy(int* S, int* Jx, int* Jy, int* Jz) {
    int e = 0;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            for (int k = 0; k < L; ++k) {
                dim3 index(i, j, k);
                e += energy(S, Jx, Jy, Jz, index, getTid(index));
            }
    printf("Total energy= %d\n", e);
}

class Configuration {
public:
    Configuration(float T, int seed = time(0)) :
            T(T) {
        beta = 1 / T;

        cudaExtent extent = make_cudaExtent(L * sizeof(int), L, L);

        cudaPitchedPtr ptrS;
        cudaPitchedPtr ptrJx;
        cudaPitchedPtr ptrJy;
        cudaPitchedPtr ptrJz;
        CUDA_CHECK_RETURN(cudaMalloc3D(&ptrS, extent));
        CUDA_CHECK_RETURN(cudaMalloc3D(&ptrJx, extent));
        CUDA_CHECK_RETURN(cudaMalloc3D(&ptrJy, extent));
        CUDA_CHECK_RETURN(cudaMalloc3D(&ptrJz, extent));
        this->ptrS = (int*) ptrS.ptr;
        this->ptrJx = (int*) ptrJx.ptr;
        this->ptrJy = (int*) ptrJy.ptr;
        this->ptrJz = (int*) ptrJz.ptr;

        blocks = dim3(BLOCKS_X, BLOCKS_X, BLOCKS_X);
        threads.x = threads.y = BLOCK_SIZE;
        threads.z = BLOCK_SIZE / 2;

        CUDA_CHECK_RETURN(cudaMalloc(&rngStates, L3 * sizeof(curandState)));

        initRNG<<<SUM_NUM_BLOCKS * 2, SUM_BLOCK_SIZE>>>(rngStates, seed);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
        fillMatrix<<<SUM_NUM_BLOCKS * 2, SUM_BLOCK_SIZE>>>(this->ptrS,
                this->ptrJx, this->ptrJy, this->ptrJz, rngStates);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());

        CUDA_CHECK_RETURN(
                cudaMalloc(&deviceSumPtr, sizeof(int) * SUM_NUM_BLOCKS));
    }

    ~Configuration() {
        CUDA_CHECK_RETURN(cudaGetLastError());

        CUDA_CHECK_RETURN(cudaFree(rngStates));
        CUDA_CHECK_RETURN(cudaFree(ptrS));
        CUDA_CHECK_RETURN(cudaFree(ptrJx));
        CUDA_CHECK_RETURN(cudaFree(ptrJy));
        CUDA_CHECK_RETURN(cudaFree(ptrJz));
        CUDA_CHECK_RETURN(cudaFree(deviceSumPtr));
        CUDA_CHECK_RETURN(cudaDeviceReset());
    }

    void nextConfig() {
        generateNext<<<blocks, threads>>>(ptrS, ptrJx, ptrJy, ptrJz, beta,
                rngStates, 0);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
        generateNext<<<blocks, threads>>>(ptrS, ptrJx, ptrJy, ptrJz, beta,
                rngStates, 1);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
    }

    double getMagnet() {
        sum<int, SUM_BLOCK_SIZE, true> <<<SUM_NUM_BLOCKS, SUM_BLOCK_SIZE,
                SUM_BLOCK_SIZE * sizeof(int)>>>((int *) ptrS, deviceSumPtr, L3);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
        CUDA_CHECK_RETURN(
                cudaMemcpy(hostSumPtr, deviceSumPtr,
                        sizeof(int) * SUM_NUM_BLOCKS, cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaGetLastError());
        double result = 0;
        for (int i = 0; i < SUM_NUM_BLOCKS; ++i) {
            result += hostSumPtr[i];
        }
        return result;
    }

    void printMatrix(int i) {
        cout << "iteration " << i << endl;
        print<<<1, 1>>>(ptrS);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
    }

    void printEnergy(int i) {
        cout << "iteration " << i << ": ";
        totalEnergy<<<1, 1>>>(ptrS, ptrJx, ptrJy, ptrJz);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
    }

private:
    mt19937 gen;
    uniform_int_distribution<> distrib;
    const float T;
    int matrix[L][L][L];
    float beta;

    curandState *rngStates;
    dim3 blocks;
    dim3 threads;
    int* ptrS;
    int* ptrJx;
    int* ptrJy;
    int* ptrJz;
    int* deviceSumPtr;
    int hostSumPtr[SUM_NUM_BLOCKS];
};

int main(int argc, char** argv) {
    double T = 0;
    unsigned int N = 82;
    if (argc >= 2)
        T = atoi(argv[1]);
    if (argc >= 3)
        N = atoi(argv[2]);
    Configuration S(T, SEED);
    double sum = 0;
    for (int i = 0; i < N; i++) {
        S.nextConfig();
        //S.printMatrix(i);
        double magnet = S.getMagnet();
        sum += magnet;
        S.printEnergy(i);
        cout << magnet << endl;
    }
    cout << sum / N << endl;
}
