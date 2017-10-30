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

#ifndef MEGA_BATCH_H
#define MEGA_BATCH_H

#include <pthread.h>
#include "mega_job.h"
#include "slabs.h"
#include "libgpuhash.h"

/* one insert buf unit */
typedef struct mega_batch_insert_s
{
	ielem_t *insert_in;
	loc_t *insert_out;
	int num_insert_job;
	pthread_mutex_t lock;
} mega_batch_insert_t;

/* for CPU */
typedef struct mega_batch_buf_s
{
	selem_t *search_in;
	selem_t *search_in_d;

	loc_t *search_out;
	loc_t *search_out_d;

	delem_t *delete_in;
	delem_t *delete_in_d;

	ielem_t *insert_in;
	ielem_t *insert_in_d;

	// Job for forwarding
	mega_job_search_t *search_job_list;
	mega_job_insert_t *insert_job_list;
	
	int num_search_job;
	int num_insert_job;
	int num_delete_job;
	int num_compact_search_job;

	mega_batch_insert_t *insert_buf;

	/* pointer array to the GPU insert bufs, this can be used by CPU */
	ielem_t **insert_in_ptrarray_h;
	/* pointer array to the GPU insert bufs, this can be used by GPU */
	void **insert_in_ptrarray_d;
	int *insert_job_num_d;
} mega_batch_buf_t;

/* Each CPU worker holds such a data structure */
typedef struct mega_batch_s
{
	mega_batch_buf_t buf[3];

	volatile int receiver_buf_id;
	volatile int sender_buf_id;
	volatile int available_buf_id[2];
	int gpu_buf_id;
	int delay;

	/* GPU worker notify CPU worker 
	 * buf_has_been_taken tell CPU worker which buf has just been taken,
	 * processed_buf_id tell CPU worker which buf has been processed.
	 * they all should be -1, if there are no events.
	 * GPU write it (0/1), and CPU clears it to -1 to claim its own action.
	 */
	pthread_mutex_t mutex_sender_buf_id; 
	pthread_mutex_t mutex_available_buf_id; 
	pthread_mutex_t mutex_batch_launch; 
	pthread_mutex_t mutex_insert_buffer;
	
	void *local_slab[MAX_NUMBER_OF_SLAB_CLASSES];
	uint32_t local_slab_size[MAX_NUMBER_OF_SLAB_CLASSES];
} mega_batch_t;

extern pthread_key_t worker_batch_struct;

#endif
