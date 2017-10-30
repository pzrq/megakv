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

//#define SELEM_NUM 3000
#define LOOP_TIME  10

int main(int argc, char *argv[])
{
	int SELEM_NUM, THREAD_NUM;
	int STREAM_NUM = 4;
	SELEM_NUM = 65536;
	THREAD_NUM = 16384;
	if (argc == 2) {
		STREAM_NUM = atoi(argv[1]);	
	} else if (argc != 3) {
		printf("usage: ./run #elem_num #thread_num\n");
	} else {
		SELEM_NUM = atoi(argv[1]);
		THREAD_NUM = atoi(argv[2]);
	}
	SELEM_NUM /= STREAM_NUM;
	printf("elem_num is %d, thread_num is %d\n", SELEM_NUM, THREAD_NUM);

	uint8_t *device_hash_table;
	uint8_t *host_hash_table;
	uint8_t *device_in[STREAM_NUM], *device_out[STREAM_NUM];
	uint8_t *host_in, *host_out[STREAM_NUM];
	double diff;
	int i, j;

	struct timespec start, end;

	cudaStream_t stream[STREAM_NUM];
	for (i = 0; i < STREAM_NUM; i ++) {
		cudaStreamCreate(&stream[i]);
	}

	cudaMalloc((void **)&(device_hash_table), HT_SIZE);
	host_hash_table = malloc(HT_SIZE);

	cudaHostAlloc((void **)&(host_in), SELEM_NUM * sizeof(selem_t), cudaHostAllocDefault);
	for (i = 0; i < STREAM_NUM; i ++) {
		cudaMalloc((void **)&(device_in[i]), SELEM_NUM * sizeof(selem_t));
		cudaMalloc((void **)&(device_out[i]), 2 * SELEM_NUM * sizeof(loc_t));
		cudaHostAlloc((void **)&(host_out[i]), 2 * SELEM_NUM * sizeof(loc_t), cudaHostAllocDefault);
	}

	srand(time(NULL));
	for (i = 0; i < (SELEM_NUM * sizeof(selem_t))/sizeof(int); i ++) {
		((int *)host_in)[i] = rand();
		for (j = 0; j < STREAM_NUM; j ++) {
			((int *)(host_out[j]))[i] = 0;
		}
	}
	for (i = 0; i < HT_SIZE/sizeof(int); i ++) {
		((int *)host_hash_table)[i] = rand();
	}

	// warm up
	cudaMemcpy(device_hash_table, host_hash_table, HT_SIZE, cudaMemcpyHostToDevice);
	for (i = 0; i < STREAM_NUM; i ++) {
		cudaMemcpyAsync(device_in[i], host_in, SELEM_NUM * sizeof(selem_t), cudaMemcpyHostToDevice, stream[i]);
		gpu_hash_search((selem_t *)device_in[i], (loc_t *)device_out[i], 
			(bucket_t *)device_hash_table, SELEM_NUM, THREAD_NUM, 128, stream[i]);
		cudaMemcpyAsync(host_out[i], device_out[i], 2 * SELEM_NUM * sizeof(loc_t), cudaMemcpyDeviceToHost, stream[i]);
	}

	// start
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (j = 0; j < LOOP_TIME; j ++) {
		for (i = 0; i < STREAM_NUM; i ++) {
			cudaMemcpyAsync(device_in[i], host_in, SELEM_NUM * sizeof(selem_t), cudaMemcpyHostToDevice, stream[i]);
			gpu_hash_search((selem_t *)device_in[i], (loc_t *)device_out[i], 
					(bucket_t *)device_hash_table, SELEM_NUM, THREAD_NUM, 128, stream[i]);
			cudaMemcpyAsync(host_out[i], device_out[i], 2 * SELEM_NUM * sizeof(loc_t), cudaMemcpyDeviceToHost, stream[i]);
		}
		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end);

	diff = 1000000 * (end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/1000;
	printf("With %d streams, Time cost for each search is %.2lf us, speed is %.2f Mops\n", 
		STREAM_NUM, (double)diff/LOOP_TIME, (double)(SELEM_NUM * STREAM_NUM * LOOP_TIME) / diff);

	return 0;
}
