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

#include "../gpu_hash.h"
#include "../libgpuhash.h"

#define LOOP_TIME 11

//GTX 480 has 14 SM, and M2090 has 16 SM
//#define INSERT_BLOCK 16 defined in gpu_hash.h
#define HASH_BLOCK_ELEM_NUM (BUC_NUM/INSERT_BLOCK)
#define BLOCK_ELEM_NUM (SELEM_NUM/INSERT_BLOCK)

#define CPU_FREQUENCY_US 2600 // 2GHz/1e6
static inline uint64_t read_tsc(void)
{
	union {
		uint64_t tsc_64;
		struct {
			uint32_t lo_32;
			uint32_t hi_32;
		};
	} tsc;
	//_mm_mfence();
	asm volatile("rdtsc" :
				"=a" (tsc.lo_32),
				"=d" (tsc.hi_32));
	return tsc.tsc_64;
}

int main(int argc, char *argv[])
{
	int SELEM_NUM;
	int STREAM_NUM = 6;
	SELEM_NUM = 65536;
	if (argc == 2) {
		STREAM_NUM = atoi(argv[1]);
	} else if (argc != 3) {
		printf("usage: ./run #stream_num #elem_num, now running with 16384\n");
	} else {
		STREAM_NUM = atoi(argv[1]);
		SELEM_NUM = atoi(argv[2]);
	}
	SELEM_NUM /= STREAM_NUM;
	int MAX_INSERT_ELEM_NUM = SELEM_NUM * 2;
	//printf("elem_num is %d, stream_num is %d\n", SELEM_NUM, STREAM_NUM);

	uint8_t *device_hash_table;
	uint8_t *device_in[STREAM_NUM];
	uint8_t *host_in[STREAM_NUM];

	ielem_t *blk_input_h[STREAM_NUM][INSERT_BLOCK];
	int	blk_elem_num_h[STREAM_NUM][INSERT_BLOCK];
	ielem_t **blk_input_d[STREAM_NUM];
	int *blk_elem_num_d[STREAM_NUM];

	double diff = 0;
	int i, j, s;

	uint64_t start, end;
	//struct  timespec kernel_start, kernel_end;
	uint8_t *device_search_in[STREAM_NUM];
	uint8_t *device_search_out[STREAM_NUM];
	uint8_t *host_search_in[STREAM_NUM];
	uint8_t *host_search_out[STREAM_NUM];
	uint8_t *host_search_verify[STREAM_NUM];

	cudaStream_t stream[STREAM_NUM];
	for (i = 0; i < STREAM_NUM; i ++) {
		cudaStreamCreate(&stream[i]);
	}

	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_hash_table), HT_SIZE));
	CUDA_SAFE_CALL(cudaMemset((void *)device_hash_table, 0, HT_SIZE));

	for (i = 0; i < STREAM_NUM; i ++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&(device_in[i]), MAX_INSERT_ELEM_NUM * sizeof(ielem_t)));
		CUDA_SAFE_CALL(cudaMemset((void *)device_in[i], 0, MAX_INSERT_ELEM_NUM * sizeof(ielem_t)));
		CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_in[i]), MAX_INSERT_ELEM_NUM * sizeof(ielem_t), cudaHostAllocDefault));

		CUDA_SAFE_CALL(cudaMalloc((void **)&(blk_input_d[i]), INSERT_BLOCK * sizeof(ielem_t *)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&(blk_elem_num_d[i]), INSERT_BLOCK * sizeof(int)));
		for (j = 0; j < INSERT_BLOCK; j ++) {
			blk_input_h[i][j] = &(((ielem_t *)device_in[i])[j*(SELEM_NUM/INSERT_BLOCK)]);
			blk_elem_num_h[i][j] = SELEM_NUM/INSERT_BLOCK;
		}

		CUDA_SAFE_CALL(cudaMemcpy(blk_input_d[i], blk_input_h[i], INSERT_BLOCK * sizeof(void *), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(blk_elem_num_d[i], blk_elem_num_h[i], INSERT_BLOCK * sizeof(int), cudaMemcpyHostToDevice));

		// for search
		CUDA_SAFE_CALL(cudaMalloc((void **)&(device_search_in[i]), SELEM_NUM * sizeof(selem_t)));
		CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_search_in[i]), SELEM_NUM * sizeof(selem_t), cudaHostAllocDefault));
		CUDA_SAFE_CALL(cudaMalloc((void **)&(device_search_out[i]), 2 * SELEM_NUM * sizeof(loc_t)));
		CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_search_out[i]), 2 * SELEM_NUM * sizeof(loc_t), cudaHostAllocDefault));
		CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_search_verify[i]), SELEM_NUM * sizeof(loc_t), cudaHostAllocDefault));
		//host_search_verify = (uint8_t *)malloc(SELEM_NUM * sizeof(loc_t));
	}

	// start
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	int loop, lower_bond;
	srand(time(NULL));

	for (loop = 0; loop < LOOP_TIME; loop ++) {

		/* +++++++++++++++++++++++++++++++++++ INSERT +++++++++++++++++++++++++++++++++ */
		for (s = 0; s < STREAM_NUM; s ++) {
			for (i = 0; i < SELEM_NUM; i += 1) {
				lower_bond = (i / BLOCK_ELEM_NUM) * HASH_BLOCK_ELEM_NUM;
				// sig
				((int *)(host_in[s]))[i*3] = rand();
				// hash
				((int *)(host_in[s]))[i*3+1] 
					= lower_bond + rand() % HASH_BLOCK_ELEM_NUM;
				// loc
				((int *)(host_in[s]))[i*3+2] = rand(); 
				//printf("%d\n", ((int *)host_search_verify)[i]);
			}
		}
		/* for debugging
		for (i = 0; i < SELEM_NUM; i += 1) {
			//printf("%d %d %d\n", ((int *)host_in)[i*3], (i/BLOCK_ELEM_NUM) * BLOCK_ELEM_NUM, 
			//(i/BLOCK_ELEM_NUM) * BLOCK_ELEM_NUM + BLOCK_ELEM_NUM);
			assert(((int *)host_in)[i*3+1] < (i/BLOCK_ELEM_NUM) * HASH_BLOCK_ELEM_NUM + HASH_BLOCK_ELEM_NUM);
			assert(((int *)host_in)[i*3+1] >= (i/BLOCK_ELEM_NUM) * HASH_BLOCK_ELEM_NUM);
		}*/


		start = read_tsc();
		for (i = 0; i < STREAM_NUM; i ++) {
#if 1
			CUDA_SAFE_CALL(cudaMemcpyAsync(device_in[i], host_in[i], MAX_INSERT_ELEM_NUM * sizeof(ielem_t), cudaMemcpyHostToDevice, stream[i]));
#else
			for (j = 0; j < INSERT_BLOCK; j ++) {
				CUDA_SAFE_CALL(cudaMemcpyAsync(blk_input_h[i][j], blk_host_h[i][j],
						(IELEM_NUM / INSERT_BLOCK) * sizeof(ielem_t), cudaMemcpyHostToDevice, stream[i]));
			}
#endif
			CUDA_SAFE_CALL(cudaMemcpyAsync(blk_elem_num_d[i], blk_elem_num_h[i], INSERT_BLOCK * sizeof(int),
						cudaMemcpyHostToDevice, stream[i]));
			gpu_hash_insert((bucket_t *)device_hash_table, 
					(ielem_t **)(blk_input_d[i]),
					(int *)(blk_elem_num_d[i]), INSERT_BLOCK, stream[i]);
		}

		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		end = read_tsc();

		if (loop > 0) {
			diff += end - start;
			//printf("%.2f\t", (double)(end - start)/(double) CPU_FREQUENCY_US);
		}
	}
	//printf("With Memcpy, speed is %.2f Mops\n",
	//	(double)(SELEM_NUM * STREAM_NUM) / ((double)(diff/(double)CPU_FREQUENCY_US)/(LOOP_TIME - 1)));
	printf("%.2f\t", (double)(diff/(double)CPU_FREQUENCY_US)/(LOOP_TIME - 1));

	return 0;
}
