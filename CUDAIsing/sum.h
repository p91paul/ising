/*
 * sum.h
 *
 *  Created on: 20/nov/2014
 *      Author: paolo
 */

#ifndef SUM_H_
#define SUM_H_

template<class T, unsigned int blockSize, bool nIsPow2>
__global__ void sum(T *g_idata, T *g_odata, unsigned int n);

#endif /* SUM_H_ */
