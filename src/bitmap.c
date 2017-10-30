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

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include "bitmap.h"


void bitmap_init(bitmap_t *bitmap, int elem_num)
{
	/* we assume there are 64X */
	assert(elem_num % BITSPERWORD == 0);
	bitmap->size = elem_num / BITSPERWORD;
	bitmap->current_size = 0;
	bitmap->walker = 0;
	bitmap->offset_mask = ~(uint64_t)0;
	bitmap->map = malloc(bitmap->size * sizeof(uint64_t));
	if (!bitmap) {
		printf("Error allocating bitmap!\n");
		exit(0);
	}
	memset((void *)bitmap->map, 0xFF, elem_num / 8);
	pthread_mutex_init(&(bitmap->lock), NULL);
}

/* If a bit is 0, it represents that this block is in use
 * if is 1, the block is free, this should be protected by
 * evict lock.
 */
uint32_t bitmap_evict(bitmap_t *bitmap)
{
	int j;

	pthread_mutex_lock(&(bitmap->lock));
	while (1) {
		/* builtin_ffsll find the first 1 bit in a 64-bit int,
		 * from the least significant bit, we use 1 to mark that
		 * this one is for eviction, and the 0s ahead should be
		 * modified to 1 for the clock algorithm (reversed version) */
		j = __builtin_ffsll(bitmap->map[bitmap->walker] & bitmap->offset_mask) - 1;
		if (j >= 0) {
			/* update the previous zeros to 1 */
			bitmap->map[bitmap->walker] |= ((uint64_t)1 << j) - 1;
			/* set this one as 0 */
			bitmap->map[bitmap->walker] &= ~((uint64_t)1 << j);
			/* update the offset_mask */
			bitmap->offset_mask = ~(((uint64_t)1 << (j + 1)) - 1);

			pthread_mutex_unlock(&(bitmap->lock));
			return bitmap->walker * BITSPERWORD + j;
		} else {
			bitmap->walker ++;
			if (bitmap->walker == bitmap->current_size)
				bitmap->walker = 0;
			bitmap->offset_mask = ~(uint64_t)0;
		}
	}
	exit(0);
}

int bitmap_evict_batch(bitmap_t *bitmap, uint32_t offset_array[], const uint32_t num)
{
	uint32_t i = 0;
	int j;
	uint64_t record;

	pthread_mutex_lock(&(bitmap->lock));

	assert(bitmap->current_size != 0 && bitmap->walker < bitmap->current_size);

	while (i < num) {
		/* evict 64 bits one time. */
		record = bitmap->map[bitmap->walker];

		while (1) {
			j = __builtin_ffsll(bitmap->map[bitmap->walker] & bitmap->offset_mask) - 1;
			if (j >= 0) {
				offset_array[i] = (bitmap->walker << BM_MASK_BIT) + j;
				i ++;

				/* update the offset_mask with (1 << 64) - 1 is wrong,
				 * since this 64 bits have been processed, break for next */
				if (j == 63) {
					break;
				} else {
					bitmap->offset_mask = ~(((uint64_t)1 << (j + 1)) - 1);
				}
			} else {
				break;
			}
		}
		bitmap->map[bitmap->walker] = ~record;

		assert(bitmap->walker < bitmap->current_size);
		bitmap->walker ++;
		if (bitmap->walker == bitmap->current_size) {
			bitmap->walker = 0;
		}
		bitmap->offset_mask = ~(uint64_t)0;
	}

	pthread_mutex_unlock(&(bitmap->lock));
	if (i == 0) {
		printf("i ==0, num is %d\n", num);
		assert(0);
	}
	return i;
}

int bitmap_update(bitmap_t *bitmap, int new_elem_num)
{
	if (bitmap->current_size + new_elem_num / BITSPERWORD >= bitmap->size) {
		return -1;
	}

	pthread_mutex_lock(&(bitmap->lock));

	/* For simplification, multiple of 64 each allocation */
	assert(new_elem_num % BITSPERWORD == 0);
	bitmap->current_size += new_elem_num / BITSPERWORD;

	assert(bitmap->current_size < bitmap->size);

	pthread_mutex_unlock(&(bitmap->lock));

	return 0;
}

int bitmap_touch(bitmap_t *bitmap, uint32_t pos)
{
	pthread_mutex_lock(&(bitmap->lock));

	/* touch the bitmap to mark the bit at @pos as 0, for the clock algorithm */
	bitmap->map[pos / BITSPERWORD] &= ~((uint64_t)1 << (pos % BITSPERWORD));

	pthread_mutex_unlock(&(bitmap->lock));

	return 0;
}
