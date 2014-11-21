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
static const int BLOCKS_XY = 6;
static const int BLOCKS_Z = 4;
static const int BLOCK_SIZE_XY = L/BLOCKS_XY;
static const int BLOCK_SIZE_Z = L / BLOCKS_Z / 2;


static const int SUM_NUM_BLOCKS = 24;
static const int SUM_BLOCK_SIZE = L3 / SUM_NUM_BLOCKS / 2;


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
