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

#ifndef _BITMAP_H_
#define _BITMAP_H_

#include <stdint.h>

#define BITSPERWORD 64
#define BM_MASK_BIT 6 // 2^6 = 64

typedef struct bitmap_s {
	uint64_t *map;
	uint32_t size; // max number of uint64_t, preallocate all of them
	uint32_t current_size; // current #items allocated
	uint32_t walker;
	uint64_t offset_mask;
	pthread_mutex_t lock;
} bitmap_t;

void bitmap_init(bitmap_t *bitmap, int elem_num);
int bitmap_touch(bitmap_t *bitmap, uint32_t pos);
int bitmap_update(bitmap_t *bitmap, int new_elem_num);
uint32_t bitmap_evict(bitmap_t *bitmap);
int bitmap_evict_batch(bitmap_t *bitmap, uint32_t offset_array[], const uint32_t num);

#endif
