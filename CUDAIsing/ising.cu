#include <curand_kernel.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <ctime>
#include <iostream>

using namespace std;
using namespace boost::random;

static const int L = 10;
static const int L3 = L * L * L;
static const int SUM_NUM_BLOCKS = 1;
static const int SUM_BLOCK_SIZE = (L3+1) / 2;
static const int N = 5;
static const double B = 0;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		cerr << "Error " << cudaGetErrorString(_m_cudaStat) << " at line "	\
				<< __LINE__ << " in file " << __FILE__ << endl;			\
		exit(1);															\
	}																		\
}

__device__ dim3 getIndex() {
	dim3 index;
	index.x = blockIdx.x * blockDim.x + threadIdx.x;
	index.y = blockIdx.y * blockDim.y + threadIdx.y;
	index.z = blockIdx.z * blockDim.z + threadIdx.z;
	return index;
}

__device__ unsigned int getTid(dim3 index){
	return index.x * L * L + index.y * L + index.z;

}

__device__ float rand(curandState * const rngStates, unsigned int tid) {
	return curand_uniform(&rngStates[tid]);
}

__device__ void tryInvert(int* S, dim3 index, float beta, curandState * const rngStates) {
	if (index.x < L && index.y < L && index.z < L) {
		//left
		index.z = (index.z + 1) % L;
		int neigh = S[getTid(index)];
		//right
		index.z = (index.z + L -2) % L;
		neigh += S[getTid(index)];
		//up
		index.z = (index.z + 1) % L;
		index.y = (index.y + 1) % L;
		neigh += S[getTid(index)];
		//down
		index.y = (index.y + L - 2) % L;
		neigh += S[getTid(index)];
		//forward
		index.y = (index.y + 1) % L;
		index.x = (index.x + 1) % L;
		neigh += S[getTid(index)];
		//backward
		index.x = (index.x + L - 2) % L;
		neigh += S[getTid(index)];
		index.x = (index.x + 1) % L;
		//energy
		unsigned int tid = getTid(index);
		int dE = -2 * S[tid] * neigh;
		if (dE < 0 || rand(rngStates, tid) < __expf(-beta * dE))
			S[getTid(index)] *= -1;
	}
}

__global__ void initRNG(curandState * const rngStates,
		const unsigned int seed) {
	unsigned int tid = getTid(getIndex());
	if (tid < L3)
		curand_init(seed, tid, 0, &rngStates[tid]);
}

__global__ void generateNext(cudaPitchedPtr Sptr, float beta,
		curandState * const rngStates) {
	int* S = (int *)Sptr.ptr;
	dim3 index = getIndex();
	index.z = 2 * index.z + (index.x % 2) ^ (index.y % 2);
	tryInvert(S, index, beta, rngStates);
	index.z++;
	tryInvert(S, index, beta, rngStates);
}

__global__ void sum(cudaPitchedPtr Sptr, int* output) {
	int* S = (int *)Sptr.ptr;
	//@@ Load a segment of the S vector into shared memory
	__shared__ int partialSum[SUM_BLOCK_SIZE * 2];
	unsigned int i = threadIdx.x, start = 2 * blockIdx.x * SUM_BLOCK_SIZE;
	if (start + i < L3)
		partialSum[i] = S[start + i];
	else
		partialSum[i] = 0;
	if (start + SUM_BLOCK_SIZE + i < L3)
		partialSum[SUM_BLOCK_SIZE + i] = S[start + SUM_BLOCK_SIZE + i];
	else
		partialSum[SUM_BLOCK_SIZE + i] = 0;
	//@@ Traverse the reduction tree
	for (unsigned int stride = SUM_BLOCK_SIZE; stride >= 1; stride >>= 1) {
		__syncthreads();
		if (i < stride)
            partialSum[i] += partialSum[i+stride];
	}
	//@@ Write the computed sum of the block to the output vector at the
	//@@ correct index
	if (i == 0)
	    output[blockIdx.x] = partialSum[0];
}

class Configuration {
public:
	Configuration(float T, int seed = time(0)) :
			T(T) {
		gen.seed(seed);
		for (int i = 0; i < L; i++)
			for (int j = 0; j < L; j++)
				for (int k = 0; k < L; k++) {
					matrix[i][j][k] = randomSpin();
				}
		beta = 1/T;
		spindist = uniform_int_distribution<>(0, 1);

		CUDA_CHECK_RETURN(cudaMalloc(&rngStates, L3 * sizeof(curandState)));

		cudaExtent extent = make_cudaExtent(L * sizeof(int), L, L);
		CUDA_CHECK_RETURN(cudaMalloc3D(&ptr, extent));

		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(matrix, L * sizeof(int), L, L);
		copyParams.dstPtr = ptr;
		copyParams.extent = extent;
		copyParams.kind = cudaMemcpyHostToDevice;
		CUDA_CHECK_RETURN(cudaMemcpy3D(&copyParams));

		threads.x = threads.y = threads.z = L;
		initRNG<<<1, threads>>>(rngStates, 5);

		 cudaMalloc(&deviceSumPtr, sizeof(int) * SUM_NUM_BLOCKS);
	}

	~Configuration(){
		CUDA_CHECK_RETURN(cudaGetLastError());

		CUDA_CHECK_RETURN(cudaFree(rngStates));
		CUDA_CHECK_RETURN(cudaFree(ptr.ptr));
		CUDA_CHECK_RETURN(cudaDeviceReset());
	}

    void nextConfig(){
    	threads.z = L/2;
    	generateNext<<<1, threads>>>(ptr, beta, rngStates);
    	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
    }

    double getMagnet(){
        sum<<<SUM_NUM_BLOCKS, SUM_BLOCK_SIZE>>>(ptr, deviceSumPtr);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        cudaMemcpy(hostSumPtr, deviceSumPtr, sizeof(int) * SUM_NUM_BLOCKS, cudaMemcpyDeviceToHost);
        double result = 0;
        for (int i = 0; i < SUM_NUM_BLOCKS; ++i) {
            result += hostSumPtr[i];
        }
        return result;
    }

private:
	mt19937 gen;
	uniform_int_distribution<> spindist;
	const float T;
	int matrix[L][L][L];
	float beta;

    curandState *rngStates;
	dim3 threads;
	cudaPitchedPtr ptr;
	int* deviceSumPtr;
	int hostSumPtr[SUM_NUM_BLOCKS];


	inline int randomSpin() {
		return 2 * spindist(gen) - 1;
	}
};

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char** argv) {
	double T;
	if (argc < 2)
		T = 0;
	else
		T = atoi(argv[1]);
	Configuration S(T, 5);
	double sum = 0;
    for (int i = 0; i<N; i++){
        S.nextConfig();
        sum += S.getMagnet();
    }
    cout << sum / N << endl;
}