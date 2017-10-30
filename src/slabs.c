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

#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/signal.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

#include "mega_config.h"
#include "mega_common.h"
#include "bitmap.h"
#include "slabs.h"
#include "items.h"

int memory_full = 0;
extern mega_config_t *config;

/* powers-of-N allocation structures */
slabclass_t slabclass[MAX_NUMBER_OF_SLAB_CLASSES];
static size_t mem_limit = 0;
static size_t mem_malloced = 0;

static void *mem_base = NULL;
static void *mem_current = NULL;
static size_t mem_avail = 0;

/**
 * Access to the slab allocator is protected by this lock
 */
static pthread_mutex_t slabs_lock = PTHREAD_MUTEX_INITIALIZER;

/*
 * Figures out which slab class (chunk size) is required to store an item of
 * a given size.
 *
 * Given object size, return id to use when allocating/freeing memory for object
 * 0 means error: can't store such a large object
 */
int slabs_clsid(const uint32_t size)
{
	int res = POWER_SMALLEST;

	if (size == 0) {
		mprint(ERROR, "size is 0 in slabs_clsid\n");
		return -1;
	}
	while (size > slabclass[res].size) {
		if (res++ == POWER_LARGEST || slabclass[res].available != 1) {
			/* won't fit in the biggest slab */
			mprint(ERROR, "size larger than biggest slab size\n");
			exit(0);
		}
	}
	return res;
}

static int grow_slab_list(const unsigned int id)
{
	slabclass_t *p = &slabclass[id];
	if (p->slabs == p->list_size) {
		uint32_t new_size =  (p->list_size != 0) ? p->list_size * 2 : 16;
		void *new_list = realloc(p->slab_list, new_size * sizeof(void *));
		if (new_list == NULL) return 0;
		p->list_size = new_size;
		p->slab_list = new_list;
	}
	return 1;
}


static void do_slabs_free(void *ptr, unsigned int id)
{
	slabclass_t *p;
	item *it = (item *)ptr;

	assert(id >= POWER_SMALLEST && id <= POWER_LARGEST);
	if (id < POWER_SMALLEST || id > POWER_LARGEST)
		return;

	p = &slabclass[id];

	it->flags = ITEM_FREE;
	it->slabs_clsid = id;
	it->next = p->slots;
	p->slots = it;

	p->sl_curr ++;
	return;
}

static void split_slab_page_into_freelist(char *ptr, const unsigned int id)
{
	slabclass_t *p = &slabclass[id];
	unsigned int x;
	for (x = 0; x < p->perslab; x ++) {
		((item *)ptr)->loc = (id << (config->loc_bits - SLAB_ID_BITS)) + (p->perslab * p->slabs) + x;
#if defined(RWLOCK)
		pthread_rwlock_init(&(((item *)ptr)->rwlock), NULL);
#endif
		do_slabs_free(ptr, id);
		ptr += p->size;
	}
}

item *slab_loc_to_ptr(const unsigned int loc)
{
	int id = loc >> (config->loc_bits - SLAB_ID_BITS);
	//int offset = loc & ~(SLAB_ID_MASK << (config->loc_bits - SLAB_ID_BITS));
	unsigned int offset = (loc << SLAB_ID_BITS) >> SLAB_ID_BITS;
	slabclass_t *p = &slabclass[id];

	assert(id <= POWER_LARGEST);
	assert(offset < p->perslab * p->slabs);
	
	//item *ptr = (item *)((char *)(p->slab_list[offset/p->perslab]) + p->size * (offset % p->perslab));
	item *ptr = (item *)((char *)(p->slab_list[offset >> p->perslab_bits])
			+ p->size * (offset & ((1 << p->perslab_bits) - 1)));
	return ptr;
}

static void *memory_allocate(uint32_t size)
{
	void *ret;

	if (mem_base == NULL) {
		/* We are not using a preallocated large memory chunk */
		ret = malloc(size);
	} else {
		ret = mem_current;

		if (size > mem_avail) {
			return NULL;
		}

		/* mem_current pointer _must_ be aligned!!! */
		if (size % ITEM_ALIGN_BYTES) {
			size += ITEM_ALIGN_BYTES - (size % ITEM_ALIGN_BYTES);
		}

		mem_current = ((char*)mem_current) + size;
		if (size < mem_avail) {
			mem_avail -= size;
		} else {
			mem_avail = 0;
		}
	}

	return ret;
}

static int do_slabs_newslab(const unsigned int id)
{
	slabclass_t *p = &slabclass[id];
	uint32_t len = p->size * p->perslab;
	char *ptr = NULL;

	if ((mem_limit && mem_malloced + len > mem_limit && p->slabs > 0) ||
		(grow_slab_list(id) == 0) ||
		((ptr = memory_allocate(len)) == 0)) {

		return 0;
	}

	memset(ptr, 0, len);
	split_slab_page_into_freelist(ptr, id);

	p->slab_list[p->slabs++] = ptr;
	mem_malloced += len;

	return 1;
}


/* Preallocate as many slab pages as possible (called from slabs_init)
   on start-up, so users don't get confused out-of-memory errors when
   they do have free (in-slab) space, but no space to make new slabs.
   if maxslabs is 18 (POWER_LARGEST - POWER_SMALLEST + 1), then all
   slab types can be made.  if max memory is less than 18 MB, only the
   smaller ones will be made.  */
static void slabs_preallocate (const unsigned int maxslabs)
{
	int i;
	unsigned int prealloc = 0;

	/* pre-allocate a 1MB slab in every size class so people don't get
	   confused by non-intuitive "SERVER_ERROR out of memory"
	   messages.  this is the most common question on the mailing
	   list.  if you really don't want this, you can rebuild without
	   these three lines.  */

	for (i = POWER_SMALLEST; i <= POWER_LARGEST; i++) {
		if (++prealloc > maxslabs)
			return;
		if (do_slabs_newslab(i) == 0) {
			fprintf(stderr, "Error while preallocating slab memory!\n"
				"If using -L or other prealloc options, max memory must be "
				"at least %d megabytes.\n", POWER_LARGEST);
			exit(1);
		}
	}

}

/**
 * Determines the chunk sizes and initializes the slab class descriptors
 * accordingly.
 */
void slabs_init(const size_t limit, const double factor, const int prealloc)
{
	int i;
	unsigned int size = sizeof(item) + 16; // add 16B, 8B minimum key and value

	mem_limit = limit;

	if (prealloc) {
		/* Allocate everything in a big chunk with malloc */
		mem_base = malloc(mem_limit);
		if (mem_base != NULL) {
			mem_current = mem_base;
			mem_avail = mem_limit;
		} else {
			mprint(ERROR, "Warning: Failed to allocate requested memory in"
					" one large chunk.\nWill allocate in smaller chunks\n");
			exit(0);
		}
	}

	memset(slabclass, 0, sizeof(slabclass));

	for (i = POWER_SMALLEST; i <= POWER_LARGEST; i ++) {
		/* Make sure items are always n-byte aligned */
		if (size % ITEM_ALIGN_BYTES) {
			size += ITEM_ALIGN_BYTES - (size % ITEM_ALIGN_BYTES);
		}

		slabclass[i].size = size;
		/* Make the number of elements in each slab a multiple of 64,
		 * this is for easy bitmap operations */
		slabclass[i].perslab_bits = config->perslab_bits;
		slabclass[i].perslab = 1 << config->perslab_bits;
		slabclass[i].slabs = 0;
		slabclass[i].slots = NULL;
		slabclass[i].sl_curr = 0;
		slabclass[i].slab_list = NULL;
		slabclass[i].list_size = 0;
		slabclass[i].available = 1;

		bitmap_init(&(slabclass[i].bitmap), config->slabclass_max_elem_num);
		size *= factor;
		mprint(INFO, "slab class %3d:   chunk size %5u,   perslab %7u\n",
					i, slabclass[i].size, slabclass[i].perslab);
	}

	//mprint(INFO, "slab class %3d: chunk size %9u perslab %7u\n",
	//			i, slabclass[i].size, slabclass[i].perslab);

	if (prealloc) {
		slabs_preallocate(POWER_LARGEST);
	}
}

void slabs_assign_size(uint32_t *size_array)
{
	int i;
	for (i = POWER_SMALLEST; 
			i <= POWER_LARGEST;
			i ++) {
		size_array[i] = slabclass[i].size;
	}
}

/* In alloc_batch,  */
static void *do_slabs_alloc_batch(const unsigned int id)
{
	slabclass_t *p = &slabclass[id];
	char *ptr = NULL, *head;
	unsigned int x;
	uint32_t len = p->size * p->perslab;

	assert(p->sl_curr == 0 || p->slots != NULL);

	if ((mem_limit && mem_malloced + len > mem_limit && p->slabs > 0) ||
		(bitmap_update(&(p->bitmap), p->perslab) == -1) ||
		(grow_slab_list(id) == 0) ||
		((head = memory_allocate(len)) == NULL)) {
		memory_full = 1;
		return NULL;
	} 

	//memset(head, 0, len);
	ptr = head;
	for (x = 0; x < p->perslab; x ++) {
		((item *)ptr)->loc = (id << (config->loc_bits - SLAB_ID_BITS)) + (p->perslab * p->slabs) + x;
		((item *)ptr)->flags = ITEM_FREE;
		((item *)ptr)->slabs_clsid = id;
		if (x == p->perslab - 1) {
			((item *)ptr)->next = NULL;
			break;
		}
		((item *)ptr)->next = ptr + p->size;
		ptr += p->size;
	}

	p->slab_list[p->slabs++] = head;
	mem_malloced += len;

	return head;
}

item *slabs_alloc_batch(uint32_t size)
{
	item *it;

	if (memory_full == 1) {
		/* never goes here */
		return NULL;
	}

	int id = slabs_clsid(size);
	if (id == -1 || id < POWER_SMALLEST || id > POWER_LARGEST) {
		mprint(ERROR, "id is wrong in slabs_alloc_batch\n");
		return NULL;
	}

	pthread_mutex_lock(&slabs_lock);
	it = do_slabs_alloc_batch(id);
	pthread_mutex_unlock(&slabs_lock);

	if (it == NULL) {
		mprint(INFO, "=========== Memory is full!! malloced %lu, available %lu ===========\n", mem_malloced, mem_avail);
		return NULL;
	}

	if (it->loc == 0) {
		mprint(DEBUG, "Ommiting the first item in the slab, because the location is 0\n");
		return it->next;
	} else {
		return it;
	}
}
