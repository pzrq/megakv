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
#include <errno.h>
#include <pthread.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <string.h>
#include <sched.h>
#include <assert.h>
#include <signal.h>

#include "mega_scheduler.h"
#include "mega_receiver.h"
#include "mega_sender.h"
#include "mega_context.h"
#include "mega_config.h"
#include "mega_memory.h"
#include "mega_macros.h"
#include "mega_job.h"
#include "mega_batch.h"
#include "mega_timer.h"
#include "mega_common.h"
#include "macros.h"

#include "libgpuhash.h"
#include <cuda_runtime.h>

extern mega_config_t *config;
extern pthread_mutex_t mutex_worker_init;

extern mega_receiver_t receivers[MAX_WORKER_NUM];
extern mega_sender_t senders[MAX_WORKER_NUM];

uint64_t search_total_job = 0, insert_total_job = 0, delete_total_job = 0;
uint64_t num_each_batch = 0;
mega_timer_t counter;
double cycles = 0;

#if defined(TIME_MEASURE)
uint64_t num_ins_del = 0;
int reset_ins_del = 0;
mega_timer_t timer_ins, timer_del, timer_search;
int stat_num_search_job = 0;
int stat_num_insert_job = 0;
int stat_num_delete_job = 0;
#endif

void handle_signal(int signal)
{
	mega_timer_stop(&counter);
	double subtime = mega_timer_get_total_time(&counter);

	printf("--------------------------------------------------\n");
	unsigned int i;
	uint64_t hit = 0, miss = 0;
	uint64_t total_packet_recv = 0, total_packet_send = 0, total_byte_recv = 0, total_byte_send = 0;

	for (i = 0; i < config->cpu_worker_num; i ++) {
		total_packet_recv += receivers[i].total_packets;
		total_byte_recv += receivers[i].total_bytes;
		total_packet_send += senders[i].total_packets;
		total_byte_send += senders[i].total_bytes;
		hit += senders[i].hit;
		miss += senders[i].miss;

		receivers[i].total_packets = 0;
		receivers[i].total_bytes = 0;
		senders[i].total_packets = 0;
		senders[i].total_bytes = 0;
		senders[i].dropped_packets = 0;
		senders[i].hit = 0;
		senders[i].miss = 0;
	}
	mprint(INFO, "Recv Packet %ld, Average len %lf, IO Speed %lfGbps.\n",
				total_packet_recv, (double)total_byte_recv/total_packet_recv,
				(double)(total_byte_recv * 8)/(double)(subtime * 1e3));
	mprint(INFO, "Send Packet %ld, Average len %lf, IO Speed %lfGbps, hit number is %ld, miss number is %ld.\n",
				total_packet_send, (double)total_byte_send/total_packet_send,
				(double)(total_byte_send * 8)/(double)(subtime * 1e3), hit, miss);

	mprint(INFO, "elapsed time : %lf us, average cycle time %lf\n \
		%ld search jobs, speed is %lf Mops,\n \
		%ld insert jobs, speed is %lf Mops,\n \
		total search and insert speed is %lf Mops,\n \
		Average batch search %lf, insert %lf, delete %lf.\n",
		subtime, cycles/((double)num_each_batch / (double)config->cpu_worker_num),
		search_total_job, (double)(search_total_job) / (double)(subtime),
		insert_total_job, (double)(insert_total_job) / (double)(subtime),
		(double)(search_total_job + insert_total_job) / (double)(subtime),
		(double)search_total_job/num_each_batch,
		(double)insert_total_job/num_each_batch,
		(double)delete_total_job/num_each_batch
		);

	search_total_job = 0;
	insert_total_job = 0;
	delete_total_job = 0;
	num_each_batch = 0;
	cycles = 0;

	mega_timer_restart(&counter);

#if defined(TIME_MEASURE)
	mprint(INFO, "insert time, num %ld, total %lf us, average %lf us, num elem %lf,\n \
		delete time, total %lf us, average %lf us, num elem %lf,\n \
		search time, total %lf us, average %lf us\n",
		num_ins_del,
		mega_timer_get_total_time(&timer_ins),
		(double)mega_timer_get_total_time(&timer_ins)/(double)num_ins_del,
		(double)stat_num_insert_job/num_ins_del,
		mega_timer_get_total_time(&timer_del),
		(double)mega_timer_get_total_time(&timer_del)/(double)num_ins_del,
		(double)stat_num_delete_job/num_ins_del,
		mega_timer_get_total_time(&timer_search),
		(double)mega_timer_get_total_time(&timer_search)/(double)num_ins_del
		);
	reset_ins_del = 1;
#endif

	return;
}

static int mega_gpu_get_available_buf_id(mega_batch_t *batch)
{
	int id;

	/* Because this is called always after mega_gpu_give_to_sender(), 
	 * There will always be at least one available buf for receiver */
	//assert(batch->available_buf_id[0] != -1);

#if defined(USE_LOCK)
	pthread_mutex_lock(&(batch->mutex_available_buf_id));
	id = batch->available_buf_id[0];
	batch->available_buf_id[0] = batch->available_buf_id[1];
	batch->available_buf_id[1] = -1; 
	pthread_mutex_unlock(&(batch->mutex_available_buf_id));
#else
	if (batch->available_buf_id[0] != -1) {
		id = batch->available_buf_id[0];
		batch->available_buf_id[0] = -1;
	} else if (batch->available_buf_id[1] != -1) {
		id = batch->available_buf_id[1];
		batch->available_buf_id[1] = -1;
	} else {
		assert(0);
	}
#endif
	return id;
}

/* Get the buffer of each CPU worker at each time interval I */
static int mega_gpu_get_batch(mega_scheduler_t *g, mega_batch_t *batch_set)
{
	int i, available_buf_id;
	mega_batch_t *batch;

	/* Tell the CPU worker we are taking the batch */
	for (i = 0; i < config->cpu_worker_num; i ++) {
		batch = &(batch_set[i]);

		assert(batch->gpu_buf_id == -1);

		available_buf_id = mega_gpu_get_available_buf_id(batch);

		batch->gpu_buf_id = batch->receiver_buf_id;

		/* Let the receiver know the new available buffer transparently */
		batch->receiver_buf_id = available_buf_id;
	}
	return batch->gpu_buf_id;
}

/* Tell the CPU sender that this batch has been completed */
static void mega_gpu_give_to_sender(mega_scheduler_t *g, mega_batch_t *batch_set, int sched_ins_del)
{
	int i;
	mega_batch_t *batch;

	for (i = 0; i < config->cpu_worker_num; i ++) {
		batch = &(batch_set[i]);

		/* Wait for the sender to complete last batch forwarding */
		while (batch->sender_buf_id != -1) ;

		if (sched_ins_del == NUM_SCHED_INS_DEL) {
			batch->delay = 0;
		} else {
			batch->delay = 1;
		}
		/* Give the buf to sender */
		batch->sender_buf_id = batch->gpu_buf_id;
		batch->gpu_buf_id = -1;
	}

	return ;
}

static int mega_scheduler_init(mega_scheduler_t *g, mega_scheduler_context_t *context)
{
	int i, j;
	mega_batch_t *batch_set = context->cpu_batch_set;

#if defined(CPU_AFFINITY)
	/* Set affinity of this gpu worker */
	unsigned long mask = 1 << context->core_id;
	if (sched_setaffinity(0, sizeof(unsigned long), (cpu_set_t *)&mask) < 0) {
		assert(0);
	}

#if 0
	/* set schedule policy */
	struct sched_param param;
	param.sched_priority = 98;
	pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
#endif
#endif

	/* set signal processing function */
	//signal(SIGINT, handle_signal);

	/* Init GPU buf set pointers */
	for (i = 0; i < 3; i ++) {
		g->bufs[i] = (mega_batch_buf_t **)malloc(config->cpu_worker_num * sizeof(mega_batch_buf_t *));
	}

	for (i = 0; i < 3; i ++) {
		for (j = 0; j < config->cpu_worker_num; j ++) {
			g->bufs[i][j] = &(batch_set[j].buf[i]);
		}
	}

	return 0;
}

/* created thread, all this calls are in the thread context */
void *mega_scheduler_main(mega_scheduler_context_t *context)
{
	mega_timer_t t, loopcounter, aa;
	int i, id = 0, gpu_buf_id, ins_id;
	double elapsed_time;
	mega_batch_buf_t *buf;
	uint8_t *device_hash_table;
	double tt;
	mega_batch_insert_t *ibuf;
	int *job_num = malloc(config->num_insert_buf * sizeof(int));
	int timeout_mark = 0;
	int sched_ins_del = NUM_SCHED_INS_DEL;
	//double time_search = 0, time_insert = 0, time_delete = 0;

	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_hash_table), HT_SIZE));
	CUDA_SAFE_CALL(cudaMemset((void *)device_hash_table, 0, HT_SIZE));

	assert(config->cpu_worker_num <= 16);
	cudaStream_t *stream = (cudaStream_t *)malloc(config->cpu_worker_num * sizeof(cudaStream_t));
	for (i = 0; i < config->cpu_worker_num; i ++) {
		cudaStreamCreate(&(stream[i]));
	}

	/* Init timers */
	mega_timer_init(&t); // For separate events
	mega_timer_init(&counter); // For the whole program
	mega_timer_init(&loopcounter); // For each loop
	mega_timer_init(&aa); // For each loop
#if defined(TIME_MEASURE)
	mega_timer_init(&timer_ins);
	mega_timer_init(&timer_del);
	mega_timer_init(&timer_search);
#endif

	/* Initialize GPU worker, we wait for that all CPU workers have been initialized
	 * then we can init GPU worker with the batches of CPU worker */
	mega_scheduler_t g;
	mega_scheduler_init(&g, context);

	mprint(INFO, "[GPU Worker] is working on core %d ...\n", context->core_id);

	/* Timers for each kernel launch */
	mega_timer_restart(&loopcounter);
	mega_timer_restart(&aa);
	
	int mark;
	i = 0;
	for (;;) {

	#if defined(TIME_MEASURE)
		if (reset_ins_del == 1) {
			num_ins_del = 0;
			mega_timer_reset(&timer_ins);
			mega_timer_reset(&timer_del);
			mega_timer_reset(&timer_search);
			reset_ins_del = 0;
			stat_num_search_job = 0;
			stat_num_insert_job = 0;
			stat_num_delete_job = 0;
		}
	#endif

		i ++;
		mark = 0;
		//////////////////////////////////////////
		/* This is a CPU/GPU synchronization point */
		do {
			mark ++;
			elapsed_time = mega_timer_get_elapsed_time(&loopcounter);
			if (elapsed_time - config->I > 10) { 
				/* surpassed the time point more than 1 us */
				mprint(DEBUG, " %d, %d -- [GPU Worker] Time point lost! : %lf, last time %lf, elapsed %lf\n",
						i, mark, elapsed_time, tt, mega_timer_get_elapsed_time(&aa));

				//timeout_mark = 1;
				#if 0
				if (mark == 1) break;

				mega_timer_stop(&counter);
				double subtime = mega_timer_get_total_time(&counter);

				printf("--------------------------------------------------\n");
				printf("elapse time : %lf us,\n \
						%ld search jobs, speed is %lf Mops,\n \
						%ld insert jobs, speed is %lf Mops,\n \
						total speed is %lf Mops,\n \
						Average batch job num for each worker is %lf, delete job number is %ld.\n",
						subtime,
						search_total_job, (double)(search_total_job) / (double)(subtime),
						insert_total_job, (double)(insert_total_job) / (double)(subtime),
						(double)(search_total_job + insert_total_job + delete_total_job) / (double)(subtime),
						(double)search_total_job/num_each_batch,
						delete_total_job);
				mega_timer_restart(&counter);
				search_total_job = 0;
				insert_total_job = 0;
				delete_total_job = 0;
				num_each_batch = 0;
				#endif
				tt = elapsed_time;
				break;
			}
			tt = elapsed_time;
		} while ((double)(config->I) - elapsed_time > 10);

		cycles += elapsed_time;

		//mprint(DEBUG, "[GPU Worker] Time point arrived : %lf\n", elapsed_time);
		//////////////////////////////////////////

		/* loopcounter is used for recording the GPU execution time for this batch/loop */
		mega_timer_restart(&loopcounter);

		/* Get Input Buffer from CPU Workers */
		gpu_buf_id = mega_gpu_get_batch(&g, context->cpu_batch_set);

		mega_timer_restart(&t);

		num_each_batch += config->cpu_worker_num;

#if defined(NOT_GPU)
		for (id = 0; id < config->cpu_worker_num; id ++) {
			buf = g.bufs[gpu_buf_id][id];
			search_total_job += buf->num_search_job;
			delete_total_job += buf->num_delete_job;
			insert_total_job += buf->num_insert_job;
		}
		goto skip_gpu;
#endif

	#if defined(TIME_MEASURE)
		mega_timer_start(&timer_search);
	#endif
		/* 1. we launch kernels for search */
		for (id = 0; id < config->cpu_worker_num; id ++) {
			buf = g.bufs[gpu_buf_id][id];

			if (buf->num_search_job == 0) {
				continue;
			} 
			search_total_job += buf->num_search_job;
	#if defined(TIME_MEASURE)
			stat_num_search_job += buf->num_search_job;
	#endif

			CUDA_SAFE_CALL(cudaMemcpyAsync(buf->search_in_d, buf->search_in, 
					buf->num_search_job * sizeof(selem_t), cudaMemcpyHostToDevice, stream[id]));
			CUDA_SAFE_CALL(cudaMemsetAsync((void *)buf->search_out_d, 0,
					2 * buf->num_search_job * sizeof(loc_t), stream[id]));
			/* launch kernel */
			gpu_hash_search(
					(selem_t *)buf->search_in_d,
					(loc_t *)buf->search_out_d,
					(bucket_t *)device_hash_table,
					buf->num_search_job,
					config->GPU_search_thread_num, 
					config->GPU_threads_per_blk,
					stream[id]);

			CUDA_SAFE_CALL(cudaMemcpyAsync(buf->search_out, buf->search_out_d, 
					2 * buf->num_search_job * sizeof(loc_t), cudaMemcpyDeviceToHost, stream[id]));
		}
	#if defined(TIME_MEASURE)
		cudaDeviceSynchronize();
		mega_timer_stop(&timer_search);
	#endif

#if (NUM_SCHED_INS_DEL > 0)
		if (sched_ins_del > 0) {
			sched_ins_del --;
			goto skip_gpu;
		} else {
			assert(sched_ins_del == 0);
			sched_ins_del = NUM_SCHED_INS_DEL;
		}
#endif

	#if defined(TIME_MEASURE)
		mega_timer_start(&timer_del);
	#endif
		/* 2. we launch kernels for delete */
		for (id = 0; id < config->cpu_worker_num; id ++) {
			buf = g.bufs[gpu_buf_id][id];

			if (buf->num_delete_job == 0) {
				continue;
			} 
			delete_total_job += buf->num_delete_job;
	#if defined(TIME_MEASURE)
			stat_num_delete_job += buf->num_delete_job;
	#endif
			assert(buf->num_delete_job < config->batch_max_delete_job);

			CUDA_SAFE_CALL(cudaMemcpyAsync(buf->delete_in_d, buf->delete_in, 
					buf->num_delete_job * sizeof(delem_t), cudaMemcpyHostToDevice, stream[id]));
			/* launch kernel */
			gpu_hash_delete(
					(delem_t *)buf->delete_in_d,
					(bucket_t *)device_hash_table,
					buf->num_delete_job,
					config->GPU_delete_thread_num, 
					config->GPU_threads_per_blk,
					stream[id]);
		}
	#if defined(TIME_MEASURE)
		cudaDeviceSynchronize();
		mega_timer_stop(&timer_del);

		mega_timer_start(&timer_ins);
	#endif


		/* 3. we launch kernels for insert */
		for (id = 0; id < config->cpu_worker_num; id ++) {
			buf = g.bufs[gpu_buf_id][id];

			if (buf->num_insert_job == 0) {
				continue;
			}
			insert_total_job += buf->num_insert_job;
	#if defined(TIME_MEASURE)
			stat_num_insert_job += buf->num_insert_job;
	#endif
			assert(buf->num_insert_job <= config->batch_max_insert_job * INSERT_BLOCK);

			for (ins_id = 0; ins_id < config->num_insert_buf; ins_id ++) {
				ibuf = &(buf->insert_buf[ins_id]);
				job_num[ins_id] = ibuf->num_insert_job;
				if (ibuf->num_insert_job == 0) {
					continue;
				}
				CUDA_SAFE_CALL(cudaMemcpyAsync((buf->insert_in_ptrarray_h)[ins_id], ibuf->insert_in,
							ibuf->num_insert_job * sizeof(ielem_t), cudaMemcpyHostToDevice, stream[id]));
			}
			CUDA_SAFE_CALL(cudaMemcpyAsync(buf->insert_job_num_d, job_num,
						config->num_insert_buf * sizeof(int), cudaMemcpyHostToDevice, stream[id]));

			/* kernel for insert */
			gpu_hash_insert((bucket_t *)device_hash_table, 
					(ielem_t **)buf->insert_in_ptrarray_d,
					(int *)buf->insert_job_num_d,
					config->num_insert_buf,
					stream[id]);
		}

		cudaDeviceSynchronize();
	#if defined(TIME_MEASURE)
		mega_timer_stop(&timer_ins);
		num_ins_del ++;
	#endif

#if (NUM_SCHED_INS_DEL > 0)
skip_gpu:
#endif
		
		if (timeout_mark == 1) {
			mega_timer_stop(&t);
			mprint(INFO, "[GPU Worker] Execution Time : %lf usi\n", 
				mega_timer_get_total_time(&t));
			/*
			mprint(INFO, "[GPU Worker] Execution Time : %lf us, ==%lf, %lf, %lf==", 
				mega_timer_get_total_time(&t), time_search, time_insert, time_delete);
			for (id = 0; id < config->cpu_worker_num; id ++) {
				buf = g.bufs[gpu_buf_id][id];
				printf("[%d, %d, %d] ", buf->num_search_job, buf->num_insert_job, buf->num_delete_job);
			}
			printf("\n");
			*/
			timeout_mark = 0;
		}

		/* Tell the senders that this batch has been processed */
		mega_gpu_give_to_sender(&g, context->cpu_batch_set, sched_ins_del);
	}

	return 0;
}
