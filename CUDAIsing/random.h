/*
 * random.h
 *
 *  Created on: 20/nov/2014
 *      Author: paolo
 */

#ifndef RANDOM_H_
#define RANDOM_H_

#include <curand_kernel.h>

__global__ void initRNG(curandState * const rngStates,
        const unsigned int seed);

__global__ void fillMatrix(int* S, curandState * const rngStates);


#endif /* RANDOM_H_ */
