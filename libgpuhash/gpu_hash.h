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

#ifndef _GPU_HASH_H_
#define _GPU_HASH_H_

#include <stdint.h>

#if defined(__CUDACC__) // NVCC
	#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
	#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
	#define MY_ALIGN(n) __declspec(align(n))
#else
	#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

typedef uint32_t sign_t;
typedef uint32_t loc_t;

typedef uint32_t hash_t;

/* In non-caching mode, the memory transaction unit is 32B, therefore, we define
the bucket size to be a multiple of 32B, meaning at least 4 elements, ELEM_NUM_P >= 2 */

#define ELEM_SIG_SIZE	8 // FIXME: sizeof(hash_t) + sizeof(sign_t)
#define ELEM_SIZE_P		3 // 2^3 bytes per element
#define ELEM_NUM_P		3 // 2^ELEM_NUM_P elements per bucket
#define ELEM_NUM		(1 << ELEM_NUM_P)
#define UNIT_THREAD_NUM_P 1
#define UNIT_THREAD_NUM (1 << UNIT_THREAD_NUM_P)


//#define MEM_P			(31) // 2^31, 2GB memory
#define MEM_P			(30) // 2^30, 1GB memory

#define BUC_P			(ELEM_NUM_P + ELEM_SIZE_P) // 2^3 is element size
/* Since 5 bits are for insert bufs, at most 32-5=27 can be used
 * as HASH_MASK. And the maximum is 1<<27 buckets, 1<<34 memory,
 * 16 GB memory */
#define HASH_MASK		((1 << (MEM_P - BUC_P)) - 1) // 2<<22 -1

#define BUC_NUM			(1 << (MEM_P - BUC_P)) 
#define HT_SIZE			(1 << (MEM_P))
  
// for insert, it is divided into 8 blocks
#define IBLOCK_P		3  // 2^3 = 8
#define INSERT_BLOCK	(1 << IBLOCK_P) // number
#define BLOCK_HASH_MASK	((1 << (MEM_P - BUC_P - IBLOCK_P)) - 1)


//#define HASH_2CHOICE	1
#define HASH_CUCKOO		1
#ifdef HASH_CUCKOO
	#define MAX_CUCKOO_NUM	5	/* maximum cuckoo evict number */
#endif


typedef MY_ALIGN(128) struct bucket_s {
	sign_t sig[ELEM_NUM];
	loc_t loc[ELEM_NUM];
} bucket_t;

// for search
typedef MY_ALIGN(8) struct selem_s {
	//uint8_t sig[ELEM_SIG_SIZE];
	sign_t	sig;
	hash_t	hash;
} selem_t;

typedef MY_ALIGN(8) union selem_shared_s {
	//uint8_t sig[ELEM_SIG_SIZE];
	selem_t elem_s;
	long long elem_u;
} selem_shared_t;

// for insert and delete
typedef struct ielem_s {
	sign_t	sig;
	hash_t	hash;
	loc_t   loc;
} ielem_t;

typedef struct ielem_s delem_t;

#endif
