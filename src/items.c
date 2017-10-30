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
#include <time.h>
#include <assert.h>

#include "bitmap.h"
#include "items.h"
#include "slabs.h"
#include "mega_config.h"
#include "mega_common.h"
#include "mega_batch.h"
#include "macros.h"

extern int memory_full;
extern mega_config_t *config;
extern slabclass_t slabclass[MAX_NUMBER_OF_SLAB_CLASSES];

item *item_alloc_batch(const uint32_t kvsize)
{
	item *it, *it_list = NULL;
	size_t ntotal = sizeof(item) + kvsize;
	int i, evict_num = 0;
	mega_batch_t *batch = pthread_getspecific(worker_batch_struct);
	mega_batch_buf_t *buf;
	uint32_t sig, hash;
	uint32_t evict_cnt = 0;
	int id = slabs_clsid(ntotal);
	if (id == -1 || id < POWER_SMALLEST || id > POWER_LARGEST) {
		mprint(ERROR, "id is wrong in slabs_alloc_batch\n");
		return NULL;
	}
	slabclass_t *p = &slabclass[id];
	int first = -1;

re:
	if (memory_full == 1) {
		first = 1;
		if (config->evict == 0) {
			assert(0);
			return NULL;
		}

		uint32_t offset, offset_array[EVICT_BATCH_SIZE + 63];

		evict_num = bitmap_evict_batch(&(p->bitmap), offset_array, config->evict_batch_size);
		assert(evict_num != 0);

		for (i = 0; i < evict_num; i ++) {
			offset = offset_array[i];

			assert(offset < p->perslab * p->slabs);
			it = (item *)((char *)(p->slab_list[offset >> p->perslab_bits])
					+ p->size * (offset & ((1 << p->perslab_bits) - 1)));
			if (it->flags != 0) {
				mprint(INFO, "Evicting an item in use\n");
				continue;
			} else {
				it->flags = ITEM_FREE;
				it->next = it_list;
				it_list = it;
				evict_cnt ++;
			}

			sig = *(uint32_t *)ITEM_key(it);
			hash = *((uint32_t *)ITEM_key(it) + 1);

again:
			id = batch->receiver_buf_id;
			buf = &(batch->buf[id]);
			if (buf->num_delete_job >=  config->batch_max_delete_job) {
				break;
			}

			(buf->delete_in[buf->num_delete_job]).hash = hash;
			(buf->delete_in[buf->num_delete_job]).sig = sig;
			(buf->delete_in[buf->num_delete_job]).loc = it->loc;
			assert(offset == ((it->loc << SLAB_ID_BITS) >> SLAB_ID_BITS));

			if (buf->num_delete_job +1 >= config->batch_max_delete_job) {
				break;
			}
			buf->num_delete_job ++;
			assert(buf->num_delete_job < config->batch_max_delete_job);

			if (id != batch->receiver_buf_id) {
				/* We do not know whether this job will be deleted successfully,
				 * so we insert again to gurantee the deletion. */
				goto again;
			}
		}

		if (evict_cnt == 0) {
			mprint(INFO, "evict cnt == 0\n");
			exit(0);
		}
	} else {
		it_list = slabs_alloc_batch(ntotal);
		first = 2;
	}

	if (it_list == NULL) {
		printf("memory_full = %d, first = %d, evict number is %d, real evict number is %d\n", memory_full, first, evict_num, evict_cnt);
		goto re;
		assert(0);
	}
	return it_list;
}
