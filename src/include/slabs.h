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

#ifndef _MEGA_SLABS_H_
#define _MEGA_SLABS_H_

#include "bitmap.h"
#include "items.h"

/* we use 8 slab classes in maximum */
#define SLAB_ID_BITS	3
#define SLAB_ID_MASK	((1 << SLAB_ID_BITS) - 1)
#define POWER_SMALLEST	0
#define POWER_LARGEST	((1 << SLAB_ID_BITS) - 1)
#define MAX_NUMBER_OF_SLAB_CLASSES (POWER_LARGEST - POWER_SMALLEST + 1) 

#define ITEM_ALIGN_BYTES 8

typedef struct {
	unsigned int size;	  /* sizes of items */
	unsigned int perslab_bits; /* related with perslab, just to reduce calculation costs */
	unsigned int perslab;   /* how many items per slab */

	void *slots;		   /* list of item ptrs */
	unsigned int sl_curr;   /* total free items in list */

	unsigned int slabs;	 /* how many slabs were allocated for this class */

	void **slab_list;	   /* array of slab pointers */
	unsigned int list_size; /* size of prev array */
	unsigned int available; /* if this class is available */

	bitmap_t bitmap; /* bitmap for clock algorithm*/

} slabclass_t;

void slabs_init(const size_t limit, const double factor, const int prealloc);
item *slab_loc_to_ptr(const unsigned int loc);
int slabs_clsid(const uint32_t size);
void slabs_assign_size(uint32_t *size_array);
item *slabs_alloc_batch(uint32_t size);

#endif
