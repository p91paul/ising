#include <ctime>
#include <iostream>

using namespace std;
//static const double B = 0;

#include <sys/time.h>
#include "common.h"
#include "sum.h"
#include "random.h"
#include "ising.h"

class Configuration {
public:
    Configuration(float T, int seed = time(0)) :
            T(T) {
        beta = 1 / T;

        cudaExtent extent = make_cudaExtent(L * sizeof(int), L, L);

        cudaPitchedPtr ptrS;
        CUDA_CHECK_RETURN(cudaMalloc3D(&ptrS, extent));
        this->ptrS = (int*) ptrS.ptr;

        blocks = dim3(BLOCKS_XY, BLOCKS_XY, BLOCKS_Z);
        threads = dim3(BLOCK_SIZE_XY, BLOCK_SIZE_XY, BLOCK_SIZE_Z);

        //cout<< blocks.x <<blocks.y << blocks.z <<endl;
        //cout << threads.x << threads.y << threads.z <<endl;

        CUDA_CHECK_RETURN(cudaMalloc(&rngStates, L3 * sizeof(curandState)));

        initRNG<<<SUM_NUM_BLOCKS * 2, SUM_BLOCK_SIZE>>>(rngStates, seed);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
        fillMatrix<<<SUM_NUM_BLOCKS * 2, SUM_BLOCK_SIZE>>>(this->ptrS,
                rngStates);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());

        CUDA_CHECK_RETURN(
                cudaMalloc(&deviceSumPtr, sizeof(int) * SUM_NUM_BLOCKS));
    }

    ~Configuration() {
        CUDA_CHECK_RETURN(cudaGetLastError());

        CUDA_CHECK_RETURN(cudaFree(rngStates));
        CUDA_CHECK_RETURN(cudaFree(ptrS));
        CUDA_CHECK_RETURN(cudaFree(deviceSumPtr));
        CUDA_CHECK_RETURN(cudaDeviceReset());
    }

    void nextConfig() {
        generateNext<false> <<<blocks, threads>>>(ptrS, beta, rngStates);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
        generateNext<true> <<<blocks, threads>>>(ptrS, beta, rngStates);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
    }

    void nextConfigGlobal() {
        generateNextGlobal<false> <<<blocks, threads>>>(ptrS, beta, rngStates);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
        generateNextGlobal<true> <<<blocks, threads>>>(ptrS, beta, rngStates);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
    }

    double getMagnet() {
        sum<int, SUM_BLOCK_SIZE, false> <<<SUM_NUM_BLOCKS, SUM_BLOCK_SIZE,
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
        totalEnergy<<<1, 1>>>(ptrS);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
    }

private:
    const float T;
    float beta;

    curandState *rngStates;
    dim3 blocks;
    dim3 threads;
    int* ptrS;
    int* deviceSumPtr;
    int hostSumPtr[SUM_NUM_BLOCKS];
};

int main(int argc, char** argv) {
    double T = 0.1;
    unsigned int N = 10;
    if (argc >= 2)
        T = atof(argv[1]);
    if (argc >= 3)
        N = atoi(argv[2]);
    Configuration S(T, SEED);
    double sum = 0;
    struct timeval t1, t2;
    double nextTime = 0, sumTime = 0;
    for (int i = 0; i < N; i++) {
        gettimeofday(&t1, NULL);
        S.nextConfig();
        gettimeofday(&t2, NULL);
        nextTime += (t2.tv_sec - t1.tv_sec) * 1000.0; // sec to ms
        nextTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
        //S.printMatrix(i);
        gettimeofday(&t1, NULL);
        double magnet = S.getMagnet();
        gettimeofday(&t2, NULL);
        sumTime += (t2.tv_sec - t1.tv_sec) * 1000.0; // sec to ms
        sumTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
        sum += magnet;
        //S.printEnergy(i);
        //cout << magnet << endl;
    }
    cout << sum / N << endl;
    cout << "Time for S.nextConfig(): " << nextTime << endl;
    cout << "Time for S.getMagnet(): " << sumTime << endl;
}
