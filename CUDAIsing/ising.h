/*
 * ising.h
 *
 *  Created on: 20/nov/2014
 *      Author: paolo
 */

#ifndef ISING_H_
#define ISING_H_

template<bool second> __global__ void generateNext(int* S, float beta, curandState * const rngStates);

__global__ void print(int* S);
__global__ void totalEnergy(int* S);



#endif /* ISING_H_ */
