/*
 * common.h
 *
 *  Created on: 20/nov/2014
 *      Author: paolo
 */

#ifndef COMMON_H_
#define COMMON_H_

static const int SEED = 5;

static const int L = 24;
static const int L3 = L * L * L;
static const int BLOCK_SIZE_X = 1;
static const int BLOCK_SIZE_Y = 4;
static const int BLOCK_SIZE_Z = 12;
static const int BLOCKS_X = L/BLOCK_SIZE_X;
static const int BLOCKS_Y = L/BLOCK_SIZE_Y;
static const int BLOCKS_Z = L/BLOCK_SIZE_Z / 2;

static const int SUM_NUM_BLOCKS = 4;
// MUST be a power of 2 for the sum to work
static const int SUM_BLOCK_SIZE = 256;
static const int SUM_SHARED_SIZE = SUM_BLOCK_SIZE > 64 ? SUM_BLOCK_SIZE : 64;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cerr << "Error " << _m_cudaStat << ": "                             \
                << cudaGetErrorString(_m_cudaStat) << " at line "           \
                << __LINE__ << " in file " << __FILE__ << endl;             \
        exit(1);                                                            \
    }                                                                       \
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T> struct SharedMemory {
    __device__ inline operator T *() {
        extern __shared__ int __smem[];
        return (T *) __smem;
    }

    __device__ inline operator const T *() const {
        extern __shared__ int __smem[];
        return (T *) __smem;
    }
};

#endif /* COMMON_H_ */
