/*
 * Copyright (c) 2015 Kai Zhang (kay21s@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>

#include "gpu_hash.h"
#include "libgpuhash.h"

#define LOOP_TIME  10
//#define KERNEL 1

int main(int argc, char *argv[])
{
	int SELEM_NUM, THREAD_NUM;
	if (argc != 3) {
		printf("usage: ./run #elem_num #thread_num\n");
		SELEM_NUM = 16384;
		THREAD_NUM = 16384;
	} else {
		SELEM_NUM = atoi(argv[1]);
		THREAD_NUM = atoi(argv[2]);
	}
	printf("SELEM_NUM = %d, THREAD_NUM = %d\n", SELEM_NUM, THREAD_NUM);

	uint8_t *device_hash_table;
	uint8_t *host_hash_table;
	uint8_t *device_in, *device_out;
	uint8_t *host_in, *host_out;
	double diff = 0;

	struct  timespec start, end;
	struct  timespec kernel_start, kernel_end;

	cudaMalloc((void **)&(device_hash_table), HT_SIZE);
	host_hash_table = malloc(HT_SIZE);

	cudaMalloc((void **)&(device_in), SELEM_NUM * sizeof(selem_t));
	cudaHostAlloc((void **)&(host_in), SELEM_NUM * sizeof(selem_t), cudaHostAllocDefault);
	cudaMalloc((void **)&(device_out), 2 * SELEM_NUM * sizeof(loc_t));
	cudaHostAlloc((void **)&(host_out), 2 * SELEM_NUM * sizeof(loc_t), cudaHostAllocDefault);

	int i;
	srand(time(NULL));
	for (i = 0; i < (SELEM_NUM * sizeof(selem_t))/sizeof(int); i ++) {
		((int *)host_in)[i] = rand();
	}
	for (i = 0; i < HT_SIZE/sizeof(int); i ++) {
		((int *)host_hash_table)[i] = rand();
	}

	// warm up
	cudaMemcpy(device_hash_table, host_hash_table, HT_SIZE, cudaMemcpyHostToDevice);

	cudaMemcpy(device_in, host_in, SELEM_NUM * sizeof(selem_t), cudaMemcpyHostToDevice);
	gpu_hash_search((selem_t *)device_in, (loc_t *)device_out, 
			(bucket_t *)device_hash_table, SELEM_NUM, THREAD_NUM, 128, 0);
	cudaMemcpy(host_out, device_out, 2 * SELEM_NUM * sizeof(loc_t), cudaMemcpyDeviceToHost);

	// start
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &start);

	for (i = 0; i < LOOP_TIME; i ++) {
		cudaMemcpy(device_in, host_in, SELEM_NUM * sizeof(selem_t), cudaMemcpyHostToDevice);

		// kernel
#if defined(KERNEL)
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &kernel_start);
#endif
		gpu_hash_search((selem_t *)device_in, (loc_t *)device_out, 
				(bucket_t *)device_hash_table, SELEM_NUM, THREAD_NUM, 128, 0);
#if defined(KERNEL)
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &kernel_end);
		diff += 1000000 * (kernel_end.tv_sec-kernel_start.tv_sec) 
			+ (double)(kernel_end.tv_nsec-kernel_start.tv_nsec)/1000;
#endif

		cudaMemcpy(host_out, device_out, 2 * SELEM_NUM * sizeof(loc_t), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}
	clock_gettime(CLOCK_MONOTONIC, &end);


#if defined(KERNEL)
	printf("Only Kernel, the difference is %.2lf us, speed is %.2f Mops\n", 
		(double)diff/LOOP_TIME, (double)(SELEM_NUM * LOOP_TIME) / diff);
#else
	diff = 1000000 * (end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/1000;
	printf("With Memcpy, the difference is %.2lf us, speed is %.2f Mops\n", 
		(double)diff/LOOP_TIME, (double)(SELEM_NUM * LOOP_TIME) / diff);
#endif

	return 0;
}
