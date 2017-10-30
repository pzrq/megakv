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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>
#include <getopt.h>

#include <signal.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include <rte_common.h>
#include <rte_log.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_memzone.h>
#include <rte_tailq.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_pci.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_ring.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>

#include "mega_context.h"
#include "mega_receiver.h"
#include "mega_sender.h"
#include "mega_scheduler.h"
#include "mega_config.h"
#include "mega_common.h"
#include "mega_memory.h"
#include "mega_macros.h"
#include "mega_job.h"
#include "mega_batch.h"
#include "mega_timer.h"
#include "mega_stat.h"
#include "slabs.h"
#include "macros.h"
#if !defined(LOCAL_TEST)
	#include "dpdk.h"
#endif

#include "libgpuhash.h"

mega_batch_t *batch_set;
mega_config_t *config;

pthread_mutex_t mutex_worker_init = PTHREAD_MUTEX_INITIALIZER;

extern mega_receiver_t receivers[MAX_WORKER_NUM];
extern mega_sender_t senders[MAX_WORKER_NUM];
extern pthread_key_t worker_batch_struct;

mega_timer_t counter;
uint8_t *device_hash_table;

stat_t stat[MAX_WORKER_NUM];

static int mega_init_config(void)
{
	config = (mega_config_t *)mega_mem_calloc(sizeof(mega_config_t));

	config->I = 200; // us

#if defined(TWO_PORTS)
	/* With 8 cores, one core is reserved as the gpu worker.
	   Therefore, with 2 ports, there can be 3 cores for each port,
	   leaving a core idle. Or there can be 3 cores for one port, and
	   4 cores for the other port to maximize the performance (TODO) */
	config->num_queue_per_port = 3;
	config->cpu_worker_num = 2 * config->num_queue_per_port;
#else
	config->num_queue_per_port = NUM_QUEUE_PER_PORT;
	config->cpu_worker_num = config->num_queue_per_port;
#endif

	config->eiu_hdr_len = 42; // eth:14 + ip:20 + udp:8 header max size
	config->ethernet_hdr_len = sizeof(struct ether_hdr); // eth:14

	config->io_batch_num = 2;
	config->ifindex0 = 0;
	config->ifindex1 = 1;

	/* TODO: remember to modify the same settings in benchmark, 
	 * or write a protocol file shared by benchmark and mega. */
	/* Specific code is needed to set bits_insert_buf to 0, because:
	 * x >> (32 - 0) = x, not 0. This is an unexpected result. */
	config->bits_insert_buf = IBLOCK_P; // 2^3 = 16
	config->num_insert_buf = INSERT_BLOCK;

	/* 100 MOPS * 300 us = 30000 jobs, when 4 workers, 8000 jobs per worker */
	config->batch_max_search_job = 32768;
	//config->batch_max_search_job = 14000;
	/* For 50% insert case, we use 1024 for each insert buf */
	//config->batch_max_insert_job = 1024;
	//config->batch_max_delete_job = 8192;

	/* per insert buf, a total of 2^IBLOCK_P bufs, make a large buf to avoid losing jobs in preloading */
	config->batch_max_insert_job = config->batch_max_search_job >> IBLOCK_P;
	config->batch_max_delete_job = config->batch_max_insert_job;

	/* There are 1<<15 items in one slab, here we use number to set slab size, not
	 * total memory. */
	config->perslab_bits = 15;
	config->slabclass_max_elem_num = 1 << 28; /* max 1 << 28 elements per slabclass */

	config->loc_bits = 32; /* 32 bits for location */
	config->slab_id_bits = 3; /* three bits for slab id, max 2^3 slabs */

	/* evict or not when memory is full? 0 - no, 1 - yes */
	config->evict = 1;
	config->evict_batch_size = EVICT_BATCH_SIZE;

	/* GPU search kernel thread num.
	 * maximum threads on a multiprocessor is 2048, and GTX780 has
	 * 12 multiprocessors, therefore totally 24576 threads can run
	 * simultaneously.
	 */
	config->GPU_search_thread_num = 24576;
	config->GPU_delete_thread_num = 16384;
	config->GPU_threads_per_blk = 256;
	config->scheduler_num = 1;

	/* maximum item size */
	config->item_max_size = 1024;

	/* maximum memory used for slab, 34->16GB */
	config->mem_limit = MEM_LIMIT;
	/* slab item size increase factor */
	config->factor = 2;
	/* 0 - not prealloc, 1 - prealloc */
	config->prealloc = 0;

	/* TODO: Get local IP and MAC address automatically */
	config->local_ip_addr = 01234;
	config->local_udp_port = 567;
	//config->local_mac_addr;

	return 0;
}

#if defined(RECEIVER_PERFORMANCE_TEST)
static void receiver_handle_signal(void)
{
	unsigned int i;
	struct timeval subtime;
	uint64_t total_rx_packets = 0, total_rx_bytes = 0;;
	mega_receiver_t *cc;
	double speed_actual = 0, mops_actual = 0;

	for (i = 0; i < config->cpu_worker_num; i ++) {
		cc = &(receivers[i]);

		gettimeofday(&(cc->endtime), NULL);
		timersub(&(cc->endtime), &(cc->startime), &(cc->subtime));
	}

	for (i = 0; i < config->cpu_worker_num; i ++) {
		cc = &(receivers[i]);
		subtime = cc->subtime;

		/* add one to avoid arithmitic error when is 0 */
		total_rx_packets = 1 + cc->total_packets;
		total_rx_bytes = cc->total_bytes;
		speed_actual += (double)(total_rx_bytes * 8) / (double) ((subtime.tv_sec * 1000000 + subtime.tv_usec) * 1000),
		mops_actual += (double)(cc->num_insert_job + cc->num_search_job) / (double) (subtime.tv_sec * 1000000 + subtime.tv_usec);
		printf("Actual   : %ld packets received, elapse time : %lds, RX Speed : %lf Mpps, %5.2f Gbps, Average Len. = %ld,\n\
				insert_job_num  %ld, speed %lf Mops, search_job_num %ld, speed %lf Mops, total speed %lf Mops\n",
				total_rx_packets, subtime.tv_sec, 
				(double)(total_rx_packets) / (double) (subtime.tv_sec * 1000000 + subtime.tv_usec),
				(double)(total_rx_bytes * 8) / (double) ((subtime.tv_sec * 1000000 + subtime.tv_usec) * 1000),
				total_rx_bytes / total_rx_packets,
				cc->num_insert_job, (double)(cc->num_insert_job) / (double) (subtime.tv_sec * 1000000 + subtime.tv_usec),
				cc->num_search_job, (double)(cc->num_search_job) / (double) (subtime.tv_sec * 1000000 + subtime.tv_usec),
				(double)(cc->num_insert_job + cc->num_search_job) / (double) (subtime.tv_sec * 1000000 + subtime.tv_usec));
	}

	printf("----------\n");
	printf("<<< actual processing speed %lf Gbps, %lf MOPS >>>\n", speed_actual, mops_actual);

	for (i = 0; i < config->cpu_worker_num; i ++) {
		cc = &(receivers[i]);
		cc->num_search_job = 0;
		cc->num_insert_job = 0;
		cc->total_packets = 0;
		cc->total_bytes = 0;
		gettimeofday(&(cc->startime), NULL);
	}

}
#endif

#if 0
static void sender_handle_signal(void)
{
	unsigned int i;
	struct timeval subtime;
	uint64_t total_tx_packets = 0, total_tx_bytes = 0;;
	mega_sender_t *cc;
	double speed_actual = 0;

	for (i = 0; i < config->cpu_worker_num; i ++) {
		cc = &(senders[i]);

		gettimeofday(&(cc->endtime), NULL);
		timersub(&(cc->endtime), &(cc->startime), &(cc->subtime));
	}

	for (i = 0; i < config->cpu_worker_num; i ++) {
		cc = &(senders[i]);
		subtime = cc->subtime;

		total_tx_packets = cc->total_packets;
		total_tx_bytes = cc->total_bytes;
		speed_actual += (double)(total_tx_bytes * 8) / (double) ((subtime.tv_sec * 1000000 + subtime.tv_usec) * 1000),
		printf("Actual: %ld packets Sent, elapse time : %lds, Send Speed : %lf Mpps, %5.2f Gbps, Aveage Len. = %ld\n", 
				total_tx_packets, subtime.tv_sec, 
				(double)(total_tx_packets) / (double) (subtime.tv_sec * 1000000 + subtime.tv_usec),
				(double)(total_tx_bytes * 8) / (double) ((subtime.tv_sec * 1000000 + subtime.tv_usec) * 1000),
				total_tx_bytes / total_tx_packets);
	}

	printf("----------\n");
	printf("<<< actual processing speed %lf >>>\n", speed_actual);

	exit(0);
}
#endif


static void mega_init_thread_keys(void)
{
	pthread_key_create(&worker_batch_struct, NULL);
}

static int mega_init_batch_set(void)
{
	/* TODO alloc the three GPU search buffers here */
	//void *device_insert_input, *device_insert_output;
	batch_set = (mega_batch_t *)mega_mem_malloc(32 * sizeof(mega_batch_t));
	return 0;
}

static int mega_init_gpu_hashtable(void)
{
	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_hash_table), HT_SIZE));
	CUDA_SAFE_CALL(cudaMemset((void *)device_hash_table, 0, HT_SIZE));

	return 0;
}

static int mega_init_stats(void)
{
	memset(stat, 0, sizeof(stat_t) * MAX_WORKER_NUM);
	return 0;
}

static int mega_launch_senders(void)
{
	unsigned int i;
	pthread_t tid;
	pthread_attr_t attr;
	mega_sender_context_t *context;

	for (i = 0; i < config->cpu_worker_num; i ++) {
		/* pass a memory block to each worker */
		context = (mega_sender_context_t *)mega_mem_malloc(sizeof(mega_sender_context_t));

#if defined(TWO_PORTS)
		if (i < config->cpu_worker_num / 2) {
			context->ifindex = config->ifindex0;
			context->queue_id = i;
		} else {
			context->ifindex = config->ifindex1;
			context->queue_id = i - config->cpu_worker_num / 2;
		}
#else
		context->ifindex = config->ifindex0;
		context->queue_id = i;
#endif
		context->unique_id = i;
		context->batch = &(batch_set[i]);

#if defined(AFFINITY_1)
		context->core_id = i + config->cpu_worker_num + 8;
#elif defined(AFFINITY_2)
		context->core_id = i + 24;
#elif defined(AFFINITY_3)
		context->core_id = i + 24;
#elif defined(AFFINITY_4)
		int start = config->cpu_worker_num % 2 == 0 ? config->cpu_worker_num+1 : config->cpu_worker_num+2;
		context->core_id = i + start;
#elif defined(AFFINITY_5)
		context->core_id = i * 2 + 13;
#endif

		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
		pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
		if (pthread_create(&tid, &attr, (void *)mega_sender_main, (void *)context) != 0) {
			printf("pthread_create error!!\n");
			return -1;
		}
	}
	return 0;
}

static int mega_launch_receivers(mega_receiver_context_t **receiver_context_set)
{
	unsigned int i;
	pthread_t tid;
	pthread_attr_t attr;
	mega_receiver_context_t *context;

	for (i = 0; i < config->cpu_worker_num; i ++) {
		/* pass a memory block to each worker */
		context = (mega_receiver_context_t *)mega_mem_malloc(sizeof(mega_receiver_context_t));
		receiver_context_set[i] = context;

#if defined(TWO_PORTS)
		if (i < config->cpu_worker_num / 2) {
			context->ifindex = config->ifindex0;
			context->queue_id = i;
		} else {
			context->ifindex = config->ifindex1;
			context->queue_id = i - config->cpu_worker_num / 2;
		}
#else
		context->ifindex = config->ifindex0;
		context->queue_id = i;
#endif
		context->unique_id = i;
		receivers[i].ifindex = context->ifindex;
		context->batch = &(batch_set[i]);

#if defined(AFFINITY_1)
		context->core_id = i + 8;
#elif defined(AFFINITY_2)
		context->core_id = i + 8;
#elif defined(AFFINITY_3)
		context->core_id = i + 8;
#elif defined(AFFINITY_4)
		int start = 1;
		context->core_id = i + start;
#elif defined(AFFINITY_5)
		context->core_id = i * 2 + 1;
#endif

		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
		pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
		if (pthread_create(&tid, &attr, (void *)mega_receiver_main, (void *)context) != 0) {
			perror("pthread_create error!!\n");
		}
	}
	return 0;
}

static int mega_launch_scheduler()
{
	pthread_t tid;
	pthread_attr_t attr;
	unsigned int i;
	mega_scheduler_context_t * context;

	assert(config->scheduler_num == 1);
	for (i = 0; i < config->scheduler_num; i ++) {
		/* pass a memory block to each worker */
		context = (mega_scheduler_context_t *)mega_mem_malloc(sizeof(mega_scheduler_context_t));
		context->cpu_batch_set = batch_set;

#if defined(AFFINITY_4)
		context->core_id = 0;
#elif defined(AFFINITY_1)
		context->core_id = 15;
#elif defined(AFFINITY_2)
		context->core_id = 31;
#else
		context->core_id = 0;
#endif

		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
		pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
		if (pthread_create(&tid, &attr, (void *)mega_scheduler_main, (void *)context) != 0) {
			perror("pthread_create error!!\n");
		}
	}
	return 0;
}

static int mega_parse_option(int argc, char*argv[])
{
	int opt, ret;
	char **argvopt;
	int option_index;
	char *prgname = argv[0];
	static struct option lgopts[] = {
		{NULL, 0, 0, 0}
	};

	argvopt = argv;

	while ((opt = getopt_long(argc, argvopt, "n:i:",
				  lgopts, &option_index)) != EOF) {

		switch (opt) {
		case 'i':
			config->I = atoi(optarg);
			printf("[ARG] Time interval is set to %d ms\n", config->I);
			break;

		case 'n':
			config->num_queue_per_port = atoi(optarg);
			printf("[ARG] %d workers in total\n", config->num_queue_per_port);
			break;
		}
	}

	if (optind >= 0)
		argv[optind-1] = prgname;

	ret = optind-1;
	optind = 0; /* reset getopt lib */
	return ret;
}

static void mega_print_arg()
{
	mprint(INFO, "=====================================================\n");

#if defined(PREFETCH_BATCH)
	mprint(INFO, "PREFETCH_BATCH enabled, DISTANCE is %d\n", PREFETCH_BATCH_DISTANCE);
#else
#if defined(PREFETCH)
	mprint(INFO, "PREFETCH enabled, DISTANCE is %d\n", PREFETCH_DISTANCE);
#else
	mprint(INFO, "PREFETCH disabled\n");
#endif
#endif

#if defined(COMPACT_JOB)
	mprint(INFO, "COMPACT_JOB enabled, ");
#else
	mprint(INFO, "COMPACT_JOB disabled, ");
#endif
#if defined(KEY_MATCH)
#if defined(COMPACT_JOB)
	mprint(ERROR, "COMPACT JOB and KEY_MATCH can not be defined at the same time\n");
	exit(0);
#endif
	mprint(INFO, "KEY_MATCH enabled\n");
#else
	mprint(INFO, "KEY_MATCH disabled\n");
#endif
#if defined(NOT_COLLECT)
	mprint(INFO, "NOT_COLLECT enabled\n");
#endif
#if defined(NOT_GPU)
	mprint(INFO, "NOT_GPU enabled\n");
#endif
#if defined(NOT_FORWARD)
	mprint(INFO, "NOT_FORWARD enabled\n");
#endif

#if defined(LOCAL_TEST)
#if defined(DIS_ZIPF)
	mprint(INFO, "DIS_ZIPF, ZIPF_THETA is %.2f\n", ZIPF_THETA);
#elif defined(DIS_UNIFORM)
	mprint(INFO, "DIS_UNIFORM, ZIPF_THETA is %.2f\n", ZIPF_THETA);
#endif
	mprint(INFO, "%d GET %d SET\n", NUM_DEFINED_GET, NUM_DEFINED_SET);
	mprint(INFO, "%d KEY_LEN, %d VALUE_LEN, LOAD FACTOR %.3f\n", 
		KEY_LEN, VALUE_LEN, LOAD_FACTOR);
#endif
	mprint(INFO, "=====================================================\n");
}

int main(int argc, char*argv[])
{
	unsigned int i, ready;
	mega_receiver_context_t *receiver_context;
	mega_init_config();
	mega_print_arg();

#if !defined(LOCAL_TEST)
	/* init EAL */
	int t_argc = 5;
	char *t_argv[] = {"./build/megakv", "-c", "ffff", "-n", "4"};
	int ret = rte_eal_init(t_argc, t_argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "Invalid EAL arguments\n");
	argc -= ret;
	argv += ret;
	mega_init_dpdk(config->num_queue_per_port);
#endif

	/* parse application arguments (after the EAL ones) */
	mega_parse_option(argc, argv);

	mega_init_batch_set();
	mega_init_stats();
	mega_init_gpu_hashtable();
	slabs_init(config->mem_limit, config->factor, config->prealloc);
	mega_init_thread_keys();

	mega_receiver_context_t **receiver_context_set;
	receiver_context_set = malloc(config->cpu_worker_num * sizeof(void *));

	//signal(SIGINT, sender_handle_signal);
	//signal(SIGINT, handle_signal);

	mega_launch_receivers(receiver_context_set);

#if !defined(RECEIVER_PERFORMANCE_TEST)
	mega_launch_senders();
	/* Synchronization, Wait for CPU workers */
	while (1) {
		ready = 0;

		pthread_mutex_lock(&mutex_worker_init);
		for (i = 0; i < config->cpu_worker_num; i ++) {
			receiver_context = receiver_context_set[i];
			if (receiver_context->initialized == 1)
				ready ++;
		}
		pthread_mutex_unlock(&mutex_worker_init);

		if (ready == config->cpu_worker_num) break;
		usleep(5000);
	}

	mega_launch_scheduler();
	printf("--------------------------------------------------\n");
#endif

	mega_timer_init(&counter);
	mega_timer_restart(&counter);

	while(1) {
		sleep(2);
#if defined(RECEIVER_PERFORMANCE_TEST)
		receiver_handle_signal();
#else
		handle_signal(0);
#endif
	}
	return 0;
}
