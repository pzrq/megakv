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

//GTX 480 has 14 SM, and M2090 has 16 SM
//#define INSERT_BLOCK 16 defined in gpu_hash.h
#define HASH_BLOCK_ELEM_NUM (BUC_NUM/INSERT_BLOCK)
#define BLOCK_ELEM_NUM (SELEM_NUM/INSERT_BLOCK)


//#define KERNEL 1

int main(int argc, char *argv[])
{
	int SELEM_NUM, THREAD_NUM;
	if (argc != 3) {
		printf("usage: ./run #elem_num #thread_num, now running with 16384\n");
		SELEM_NUM = 16384;
		THREAD_NUM = 16384;
	} else {
		SELEM_NUM = atoi(argv[1]);
		THREAD_NUM = atoi(argv[2]);
	}
	printf("elem_num is %d, thread_num is %d\n", SELEM_NUM, THREAD_NUM);

	uint8_t *device_hash_table;
	uint8_t *device_in;
	uint8_t *host_in;

	uint8_t *host_hash_table;
	host_hash_table = malloc(HT_SIZE);

	ielem_t *blk_input_h[INSERT_BLOCK];
	int	blk_elem_num_h[INSERT_BLOCK];
	ielem_t **blk_input_d;
	int *blk_elem_num_d;

	double diff;
	int i;

	struct  timespec start, end;
#if defined(KERNEL)
	struct  timespec kernel_start;
#endif
	uint8_t *device_search_in;
	uint8_t *device_search_out;
	uint8_t *host_search_in;
	uint8_t *host_search_out;
	uint8_t *host_search_verify;

	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_hash_table), HT_SIZE));
	CUDA_SAFE_CALL(cudaMemset((void *)device_hash_table, 0, HT_SIZE));

	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_in), SELEM_NUM * sizeof(ielem_t)));
	CUDA_SAFE_CALL(cudaMemset((void *)device_in, 0, SELEM_NUM * sizeof(ielem_t)));
	CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_in), SELEM_NUM * sizeof(ielem_t), cudaHostAllocDefault));

	CUDA_SAFE_CALL(cudaMalloc((void **)&(blk_input_d), INSERT_BLOCK * sizeof(ielem_t *)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&(blk_elem_num_d), INSERT_BLOCK * sizeof(int)));
	for (i = 0; i < INSERT_BLOCK; i ++) {
		blk_input_h[i] = &(((ielem_t *)device_in)[i*(SELEM_NUM/INSERT_BLOCK)]);
		blk_elem_num_h[i] = SELEM_NUM/INSERT_BLOCK;
	}

	CUDA_SAFE_CALL(cudaMemcpy(blk_input_d, blk_input_h, INSERT_BLOCK * sizeof(void *), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(blk_elem_num_d, blk_elem_num_h, INSERT_BLOCK * sizeof(int), cudaMemcpyHostToDevice));

	// for search
	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_search_in), SELEM_NUM * sizeof(selem_t)));
	CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_search_in), SELEM_NUM * sizeof(selem_t), cudaHostAllocDefault));
	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_search_out), 2 * SELEM_NUM * sizeof(loc_t)));
	CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_search_out), 2 * SELEM_NUM * sizeof(loc_t), cudaHostAllocDefault));
	CUDA_SAFE_CALL(cudaHostAlloc((void **)&(host_search_verify), SELEM_NUM * sizeof(loc_t), cudaHostAllocDefault));
	//host_search_verify = (uint8_t *)malloc(SELEM_NUM * sizeof(loc_t));

	// start
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	int has, lower_bond;
	srand(time(NULL));

	for (has = 0; has < 0.1 * HT_SIZE/(sizeof(sign_t) + sizeof(loc_t)); has += SELEM_NUM) {

		printf("%d : Load factor: %f, exisiting number : %d.\n", has/SELEM_NUM, 
				(double)has/(HT_SIZE/(sizeof(sign_t)+sizeof(loc_t))), has);
		/* +++++++++++++++++++++++++++++++++++ INSERT +++++++++++++++++++++++++++++++++ */
		for (i = 0; i < SELEM_NUM; i += 1) {
			lower_bond = (i / BLOCK_ELEM_NUM) * HASH_BLOCK_ELEM_NUM;
			// sig
			((int *)host_search_in)[i*2]
				= ((int *)host_in)[i*3] = rand();
			// hash
			((int *)host_search_in)[i*2+1] 
				= ((int *)host_in)[i*3+1] 
				= lower_bond + rand() % HASH_BLOCK_ELEM_NUM;
			// loc
			((int *)host_search_verify)[i]
				= ((int *)host_in)[i*3+2] = rand(); 
			//printf("%d\n", ((int *)host_search_verify)[i]);
		}
		/* for debugging
		for (i = 0; i < SELEM_NUM; i += 1) {
			//printf("%d %d %d\n", ((int *)host_in)[i*3], (i/BLOCK_ELEM_NUM) * BLOCK_ELEM_NUM, 
			//(i/BLOCK_ELEM_NUM) * BLOCK_ELEM_NUM + BLOCK_ELEM_NUM);
			assert(((int *)host_in)[i*3+1] < (i/BLOCK_ELEM_NUM) * HASH_BLOCK_ELEM_NUM + HASH_BLOCK_ELEM_NUM);
			assert(((int *)host_in)[i*3+1] >= (i/BLOCK_ELEM_NUM) * HASH_BLOCK_ELEM_NUM);
		}*/


		clock_gettime(CLOCK_MONOTONIC, &start);
		CUDA_SAFE_CALL(cudaMemcpy(device_in, host_in, SELEM_NUM * sizeof(ielem_t), cudaMemcpyHostToDevice));
#if defined(KERNEL)
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &kernel_start);
#endif
		gpu_hash_insert((bucket_t *)device_hash_table, 
				(ielem_t **)blk_input_d,
				(int *)blk_elem_num_d, INSERT_BLOCK, 0);
		//cudaDeviceSynchronize();
		//clock_gettime(CLOCK_MONOTONIC, &kernel_end);

		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC, &end);

#if defined(KERNEL)
		diff = 1000000 * (end.tv_sec-kernel_start.tv_sec) + (double)(end.tv_nsec-kernel_start.tv_nsec)/1000;
		printf("Insert kernel, the difference is %.2lf us, speed is %.2f Mops\n", 
			(double)diff, (double)(SELEM_NUM) / diff);
#else
		diff = 1000000 * (end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/1000;
		printf("With Memcpy, the difference is %.2lf us, speed is %.2f Mops\n", 
			(double)diff, (double)(SELEM_NUM) / diff);
#endif


#if 1
		/* +++++++++++++++++++++++++++++++++++ SEARCH +++++++++++++++++++++++++++++++++ */
		CUDA_SAFE_CALL(cudaMemcpy(host_hash_table, device_hash_table, HT_SIZE, cudaMemcpyDeviceToHost));

		// verify with search
		CUDA_SAFE_CALL(cudaMemcpy(device_search_in, host_search_in, 
				SELEM_NUM * sizeof(selem_t), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemset((void *)device_search_out, 0, 2 * SELEM_NUM * sizeof(loc_t)));
		gpu_hash_search((selem_t *)device_search_in, (loc_t *)device_search_out, 
				(bucket_t *)device_hash_table, SELEM_NUM, THREAD_NUM, 128, 0);
		CUDA_SAFE_CALL(cudaMemcpy(host_search_out, device_search_out, 
				2 * SELEM_NUM * sizeof(loc_t), cudaMemcpyDeviceToHost));

		for (i = 0; i < SELEM_NUM; i ++) {
			if(((int *)host_search_out)[i<<1] != ((int *)host_search_verify)[i] 
				&& ((int *)host_search_out)[(i<<1)+1] != ((int *)host_search_verify)[i]) {
				printf("not found insertion %d : out %d and %d, should be : %d\n", i,
						((int *)host_search_out)[i<<1], ((int *)host_search_out)[(i<<1)+1],
						((int *)host_search_verify)[i]);
				/* for debugging
				((int *)host_in)[0] = ((int *)host_in)[i*3];
				((int *)host_in)[1] = ((int *)host_in)[i*3+1];
				((int *)host_in)[2] = ((int *)host_in)[i*3+2];
				CUDA_SAFE_CALL(cudaMemcpy(device_in, host_in, sizeof(ielem_t), cudaMemcpyHostToDevice));
				CUDA_SAFE_CALL(cudaMemset((void *)device_out, 0, SELEM_NUM * sizeof(loc_t)));
				gpu_hash_insert((bucket_t *)device_hash_table, 
					(ielem_t **)blk_input_d, (loc_t **)blk_output_d,
					(int *)blk_elem_num_d, INSERT_BLOCK, 0);
				*/
			}
		}
#endif

	
		/* +++++++++++++++++++++++++++++++++++ DELETE +++++++++++++++++++++++++++++++++ */
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC, &start);
		CUDA_SAFE_CALL(cudaMemcpy(device_in, host_in, SELEM_NUM * sizeof(delem_t), cudaMemcpyHostToDevice));
#if defined(KERNEL)
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &kernel_start);
#endif
		gpu_hash_delete(
				(delem_t *)device_in,
				(bucket_t *)device_hash_table, 
				SELEM_NUM, THREAD_NUM, 128, 0);
		//cudaDeviceSynchronize();
		//clock_gettime(CLOCK_MONOTONIC, &kernel_end);

		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC, &end);

#if defined(KERNEL)
		diff = 1000000 * (end.tv_sec-kernel_start.tv_sec) + (double)(end.tv_nsec-kernel_start.tv_nsec)/1000;
		printf("DELETE kernel, the difference is %.2lf us, speed is %.2f Mops\n", 
			(double)diff, (double)(SELEM_NUM) / diff);
#else
		diff = 1000000 * (end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/1000;
		printf("DELETE, the difference is %.2lf us, speed is %.2f Mops\n", 
			(double)diff, (double)(SELEM_NUM) / diff);
#endif


		// verify with search
		CUDA_SAFE_CALL(cudaMemcpy(device_search_in, host_search_in, 
				SELEM_NUM * sizeof(selem_t), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemset((void *)device_search_out, 0, 2 * SELEM_NUM * sizeof(loc_t)));
		gpu_hash_search((selem_t *)device_search_in, (loc_t *)device_search_out, 
				(bucket_t *)device_hash_table, SELEM_NUM, THREAD_NUM, 128, 0);
		CUDA_SAFE_CALL(cudaMemcpy(host_search_out, device_search_out, 
				2 * SELEM_NUM * sizeof(loc_t), cudaMemcpyDeviceToHost));

		for (i = 0; i < SELEM_NUM; i ++) {
			if(((int *)host_search_out)[i<<1] == ((int *)host_search_verify)[i] 
				|| ((int *)host_search_out)[(i<<1)+1] == ((int *)host_search_verify)[i]) {
				printf("found insertion %d : out %d and %d, should be : %d\n", i,
						((int *)host_search_out)[i<<1], ((int *)host_search_out)[(i<<1)+1],
						((int *)host_search_verify)[i]);
			}
		}
	}

	//diff = 1000000 * (kernel_end.tv_sec-kernel_start.tv_sec) 
	//	+ (double)(kernel_end.tv_nsec-kernel_start.tv_nsec)/1000;
	//printf("Only Kernel, the difference is %.2lf us, speed is %.2f Mops\n", 
	//	(double)diff, (double)(SELEM_NUM) / diff);

	return 0;
}
