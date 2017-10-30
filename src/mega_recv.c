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
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <errno.h>
#include <stdint.h>
#include <sched.h>
#include <sys/time.h>

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
#include <rte_ip.h>
#include <rte_byteorder.h>
#include <rte_udp.h>
#include <rte_ethdev.h>
#include <rte_ring.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>

#include "mega_context.h"
#include "mega_receiver.h"
#include "mega_config.h"
#include "mega_memory.h"
#include "mega_macros.h"
#include "mega_job.h"
#include "mega_batch.h"
#include "mega_common.h"
#include "mega_stat.h"

#include "libgpuhash.h"
#include "macros.h"
#include "zipf.h"

#if defined(RECEIVER_PERFORMANCE_TEST)
#undef COMPACT_JOB 
#endif

#define BATCH_ALLOC(it, size, batch) \
	do { \
		it = NULL; \
		for (i = POWER_SMALLEST; i < POWER_LARGEST; i ++) { \
			if (size <= batch->local_slab_size[i]) { \
				break; \
			} \
		} \
		if (i == MAX_NUMBER_OF_SLAB_CLASSES) { \
			mprint(ERROR, "size larger than biggest slab size\n"); \
			break; \
		} \
		if (batch->local_slab[i] == NULL) { \
			batch->local_slab[i] = item_alloc_batch(size); \
		} \
		assert(batch->local_slab[i] != NULL); \
		it = (item *)(batch->local_slab[i]); \
		batch->local_slab[i] = it->next; \
		it->flags = 0; \
	} while(0)

#if 0
	/* prevent item being evicted during insertion */
		it->flags = ITEM_WRITING;
#endif



pthread_key_t worker_batch_struct;
pthread_key_t receiver;

extern mega_config_t *config;
extern pthread_mutex_t mutex_worker_init;
extern stat_t stat[MAX_WORKER_NUM];

mega_receiver_t receivers[MAX_WORKER_NUM];
#if defined(PRELOAD)
int loading_mode = 1;
#endif

static int mega_receiver_batch_init(void)
{
	unsigned int i, j;

	mega_batch_t *batch = pthread_getspecific(worker_batch_struct);
	void *device_search_input, *device_search_output;
	void *device_delete_input, *device_insert_input;
	mega_batch_insert_t *p;
	void **host_insert_in_ptrarray;
	void **device_insert_in_ptrarray;
	void *device_insert_job_num;

	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_search_input), config->batch_max_search_job * sizeof(selem_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_search_output), config->batch_max_search_job * sizeof(loc_t) * 2));

	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_delete_input), config->batch_max_delete_job * sizeof(delem_t)));

	/* The insert array pointers in GPU and host */
	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_insert_in_ptrarray), config->num_insert_buf * sizeof(void *)));
	host_insert_in_ptrarray = malloc(config->num_insert_buf * sizeof(void *));

	/* init one set of device buffers for the gpu worker */
	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_insert_input), config->batch_max_insert_job * config->num_insert_buf * sizeof(ielem_t)));
	for (i = 0; i < config->num_insert_buf; i ++) {
		host_insert_in_ptrarray[i] = (char *)device_insert_input + config->batch_max_insert_job * i * sizeof(ielem_t);
	}

	/* copy the input&output pointers to GPU only once, we don't need to do this later */
	CUDA_SAFE_CALL(cudaMemcpy(device_insert_in_ptrarray, host_insert_in_ptrarray, 
				config->num_insert_buf * sizeof(void *), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void **)&(device_insert_job_num), config->num_insert_buf * sizeof(int)));

	for (i = 0; i < 3; i ++) {
		/* Allocate pinned memory: 0x00 -- cudaHostAllocDefault, 0x04 -- cudaHostAllocWriteCombined */
		CUDA_SAFE_CALL(cudaHostAlloc((void **)&(batch->buf[i].search_in), config->batch_max_search_job * sizeof(selem_t), 0x00));
		CUDA_SAFE_CALL(cudaHostAlloc((void **)&(batch->buf[i].search_out), config->batch_max_search_job * sizeof(loc_t) * 2, 0x00));
		CUDA_SAFE_CALL(cudaHostAlloc((void **)&(batch->buf[i].delete_in), config->batch_max_delete_job * sizeof(delem_t), 0x00));

		memset(batch->buf[i].search_in, 0, config->batch_max_search_job * sizeof(selem_t));
		batch->buf[i].search_in_d = (selem_t *)device_search_input;
		batch->buf[i].search_out_d = (loc_t *)device_search_output;
		batch->buf[i].delete_in_d = (delem_t *)device_delete_input;
		batch->buf[i].insert_in_d = (delem_t *)device_insert_input;
		
		batch->buf[i].search_job_list = mega_mem_malloc(config->batch_max_search_job * sizeof(mega_job_search_t));
		batch->buf[i].insert_job_list = mega_mem_malloc(config->batch_max_insert_job * config->num_insert_buf * sizeof(mega_job_insert_t));
		batch->buf[i].num_search_job = 0;
		batch->buf[i].num_insert_job = 0;
		batch->buf[i].num_delete_job = 0;
		batch->buf[i].num_compact_search_job = 0;


		/* Init insert bufs, one buf will have config->num_insert_buf host buffers */
		p = (mega_batch_insert_t *)calloc(config->num_insert_buf, sizeof(mega_batch_insert_t));	

		/* allocate one set insert bufs */
		CUDA_SAFE_CALL(cudaHostAlloc((void **)&(batch->buf[i].insert_in), config->batch_max_insert_job * config->num_insert_buf * sizeof(ielem_t), 0x00));
		for (j = 0; j < config->num_insert_buf; j ++) {
			//CUDA_SAFE_CALL(cudaHostAlloc((void **)&(p[j].insert_in), config->batch_max_insert_job * sizeof(ielem_t), 0x00));
			p[j].insert_in = (ielem_t *)((char *)(batch->buf[i].insert_in) + config->batch_max_insert_job * j * sizeof(ielem_t));
			p[j].num_insert_job = 0;
			//pthread_mutex_init(&(p[j].lock), NULL);
		}

		/* set the insert bufs */
		batch->buf[i].insert_buf = p;
		batch->buf[i].insert_job_num_d = device_insert_job_num;

		/* Pointers to GPU bufs, this ptr array can be read by host */
		batch->buf[i].insert_in_ptrarray_h = (ielem_t **)host_insert_in_ptrarray;
		/* Pointers to GPU bufs, this ptr array can be read by device */
		batch->buf[i].insert_in_ptrarray_d = device_insert_in_ptrarray;
	}


	batch->sender_buf_id = -1;
	batch->gpu_buf_id = -1;
	batch->delay = -1;
	/* The receiver buffer currently using is #0 */
	batch->receiver_buf_id = 0;
	/* At first the available buf is #1 and #2 */
	batch->available_buf_id[0] = 1;
	batch->available_buf_id[1] = 2;
	
	assert(pthread_mutex_init(&(batch->mutex_sender_buf_id), NULL) == 0);
	assert(pthread_mutex_init(&(batch->mutex_available_buf_id), NULL) == 0);
	assert(pthread_mutex_init(&(batch->mutex_batch_launch), NULL) == 0);

	memset(batch->local_slab, 0, sizeof(void *) * MAX_NUMBER_OF_SLAB_CLASSES);
	memset(batch->local_slab_size, 0, sizeof(uint32_t) * MAX_NUMBER_OF_SLAB_CLASSES);
	slabs_assign_size(batch->local_slab_size);

	return 0;
}

static int mega_receiver_init(mega_receiver_context_t *context)
{
	unsigned int i;

	mega_batch_t *batch = context->batch;
	pthread_setspecific(worker_batch_struct, (void *)batch);
	__builtin_prefetch(batch);
	__builtin_prefetch(&worker_batch_struct); 

	for (i = 0; i < config->cpu_worker_num; i ++) {
		receivers[i].total_packets = 0;
		receivers[i].total_bytes = 0;
		receivers[i].num_search_job = 0;
		receivers[i].num_insert_job = 0;
	}
	
	/* Init receiver batch */
	mega_receiver_batch_init();

#if defined(CPU_AFFINITY)
	/* set schedule affinity */
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

	pthread_mutex_lock(&mutex_worker_init);
	context->initialized = 1;
	pthread_mutex_unlock(&mutex_worker_init);

	return 0;
}

int process_packet(char *pkt_ptr, int pkt_len, int unique_id)
{
	char *payload_ptr;
	uint32_t ip;
	uint16_t port;
	int payload_len;
	mega_receiver_t *cc = &(receivers[unique_id]);

	/* Parser packet header 
	 * ----------------------------------------------------- */
	struct ether_hdr *ethh = (struct ether_hdr *)pkt_ptr;
	struct ipv4_hdr *iph = (struct ipv4_hdr *)(ethh + 1);
	ip = iph->src_addr;
	struct udp_hdr *udph = (struct udp_hdr *)(iph + 1);
	payload_ptr = (char *)(udph + 1);
	payload_len = pkt_len - (payload_ptr - pkt_ptr);
	port = udph->src_port;

	/* Parser megakv 
	 * ----------------------------------------------------- */
	uint16_t job_type;
	char *key;
	char *ptr = payload_ptr;
	uint16_t nkey;
	uint32_t nbyte;
	int id, i;
	uint64_t sig;
#if defined(SIGNATURE)
	uint64_t mask64;
#endif
	uint32_t hash, sig1, insert_block_id;
	int kk = 0;
	item *it;

	mega_batch_t *batch = pthread_getspecific(worker_batch_struct);

	/* Judge if this is a right packet, and skip header */
	if (*(uint16_t *)ptr != PROTOCOL_MAGIC) {
		mprint(ERROR, "magic number wrong\n");
		return -1;
	}
	ptr += MEGA_MAGIC_NUM_LEN;

	/* add jobs */
	for (;;) {
		job_type = *(uint16_t *)ptr;
		ptr += MEGA_JOB_TYPE_LEN;

		switch (job_type) {
		case 0xFFFF:
			/* end of packet, return from here */
			return kk;

		case MEGA_JOB_GET:
			nkey = *(uint16_t *)ptr;
			ptr += MEGA_KEY_SIZE_LEN;

			do {
				if (nkey != MAX_KEY_LENGTH) {
					mprint(ERROR, "key length wrong: %d\n", nkey);
					break;
				}
again:
				id = batch->receiver_buf_id;
				mega_batch_buf_t *buf = &(batch->buf[id]);
				if (buf->num_search_job >= config->batch_max_search_job - 1) {
					mprint(DEBUG, "exceed max job num %d > %d\n", buf->num_search_job, config->batch_max_search_job);
					return -1;
				}

				cc->num_search_job ++;
				kk ++;

				mega_job_search_t *job = &(buf->search_job_list[buf->num_compact_search_job]);

#if defined(COMPACT_JOB)
				if (job->ip == ip && job->port == port) {
					/* the general info of this job is the same with previous one */
					job->num ++;
				} else {
					/* Not the same, we use a new job struct */
			#if !defined(RECEIVER_PERFORMANCE_TEST)
					buf->num_compact_search_job ++;
			#endif
					mega_job_search_t *job = &(buf->search_job_list[buf->num_compact_search_job]);
					job->num = 1;
					job->eth_addr = ethh->s_addr;
					job->ip = ip;
					job->port = port;
					/* index in the search buffer */
					job->index = buf->num_search_job;
				}

				key = ptr;
				sig = *(uint64_t *)key;
			#if defined(SIGNATURE)
				/* Use XOR/AES to calculate hash and signature */
				for (i = 8; i <= nkey - 8; i += 8) {
					sig = sig ^ *(uint64_t *)(key + i);
				}
				if (i < nkey) {
					mask64 = ((uint64_t)1 << ((nkey - i) * 8)) - 1;
					sig = sig ^ (*(uint64_t *)(key + i) & mask64);
				}
			#endif
				(buf->search_in[buf->num_search_job]).hash = (uint32_t)(sig >> 32);
				(buf->search_in[buf->num_search_job]).sig = (uint32_t)((sig << 32) >> 32);

			#if !defined(RECEIVER_PERFORMANCE_TEST)
				buf->num_search_job ++;
			#endif

				/* If the buffer is swapped during the process, we do not know
				 * whether the job has been successfully added to the batch,
				 * we just do job_num--, so that the Sender will ignore it,
				 * this is a lock-free optimistic approach.
				 * If it is not successfully added, the info will be wrong,
				 * and GPU may not find it. Even it is found, the client will
				 * ignore it. */
				if (id != batch->receiver_buf_id) {
					job->num --;
					if (job->num == 0)	buf->num_compact_search_job --;
					if (buf->num_compact_search_job < 0) buf->num_compact_search_job = 0;
					buf->num_search_job --;
					if (buf->num_search_job < 0) buf->num_search_job = 0;
					goto again;
				}

#else /* not defined COMPACT_JOB */
				job->eth_addr = ethh->s_addr;
				job->ip = ip;
				job->port = port;
				/* index in the search buffer */
				job->index = buf->num_search_job;
				job->num = 1;

				key = ptr;
				sig = *(uint64_t *)key;
			#if defined(SIGNATURE)
				/* Use XOR/AES to calculate hash and signature */
				for (i = 8; i <= nkey - 8; i += 8) {
					sig = sig ^ *(uint64_t *)(key + i);
				}
				if (i < nkey) {
					mask64 = ((uint64_t)1 << ((nkey - i) * 8)) - 1;
					sig = sig ^ (*(uint64_t *)(key + i) & mask64);
				}
			#endif
				(buf->search_in[buf->num_search_job]).hash = (uint32_t)(sig >> 32);
				(buf->search_in[buf->num_search_job]).sig = (uint32_t)((sig << 32) >> 32);

			#if defined(KEY_MATCH)
				job->nkey = nkey;
				rte_memcpy(job->key, key, nkey);
			#endif

			#if !defined(RECEIVER_PERFORMANCE_TEST)
				buf->num_compact_search_job ++;
				buf->num_search_job ++;
			#endif

				if (id != batch->receiver_buf_id) {
					/* It may be wrong if the GPU worker and Sender is processing too fast.
					 * After the switching, the GPU worker handles to Sender, and Sender clears
					 * num_search_job to 0, then a minus 1 operation would make it very huge.
					 * This will only happen when disabling both GPU Worker and Sender. */
					buf->num_compact_search_job --;
					if (buf->num_compact_search_job < 0) buf->num_compact_search_job = 0;
					buf->num_search_job --;
					if (buf->num_search_job < 0) buf->num_search_job = 0;
					goto again;
				}

#endif /* COMPACT JOB */

			} while(0);

			ptr += nkey;
			break;

		case MEGA_JOB_SET:
			/* Get key length */
			nkey = *(uint16_t *)ptr;
			ptr += MEGA_KEY_SIZE_LEN;

			/* Get value length */
			nbyte = *(uint32_t *)ptr;
			ptr += MEGA_VALUE_SIZE_LEN;
			key = ptr;

			BATCH_ALLOC(it, nkey + nbyte, batch);
			if (it == NULL) {
				ptr += (nkey + nbyte);
				break;
			}
			it->nkey = nkey;
			it->nbytes = nbyte;
			rte_memcpy(ITEM_key(it), key, nkey + nbyte);

			do {
				if (nkey != MAX_KEY_LENGTH || nbyte != MAX_VALUE_LENGTH) {
					it->flags = 0;
					mprint(ERROR, "key or value length wrong: %d, %d\n", nkey, nbyte);
					break;
				}

				sig = *(uint64_t *)key;
			#if defined(SIGNATURE)
				/* Use XOR/AES to calculate hash and signature */
				for (i = 8; i <= nkey - 8; i += 8) {
					sig = sig ^ *(uint64_t *)(key + i);
				}
				if (i < nkey) {
					mask64 = ((uint64_t)1 << ((nkey - i) * 8)) - 1;
					sig = sig ^ (*(uint64_t *)(key + i) & mask64);
				}
			#endif
				hash = (uint32_t)(sig >> 32);
				sig1 = (uint32_t)((sig << 32) >> 32);

				if (config->bits_insert_buf == 0)	insert_block_id = 0;
				else	insert_block_id = (uint32_t)(hash >> (32 - config->bits_insert_buf));

				cc->num_insert_job ++;
				kk ++;

reinsert:
				id = batch->receiver_buf_id;
				mega_batch_buf_t *buf = &(batch->buf[id]);
				if (buf->num_insert_job >= ((config->batch_max_insert_job - 1) << config->bits_insert_buf)) {
					it->flags = 0;
					mprint(DEBUG, "insert exceed max job num %d > %d, ( %d, %d )\n", buf->num_insert_job,
						((config->batch_max_insert_job - 1) << config->bits_insert_buf),
						buf->insert_buf[0].num_insert_job, buf->insert_buf[1].num_insert_job);
					return -1;
				}

				mega_batch_insert_t *insert_buf = &(buf->insert_buf[insert_block_id]);
				//assert(insert_buf->num_insert_job < ((1 << (32 - config->bits_insert_buf)) - 1));
				if (insert_buf->num_insert_job  + 1 >= config->batch_max_insert_job) {
					it->flags = 0;
					mprint(DEBUG, "insert exceed max insert job num %d >= %d\n", insert_buf->num_insert_job, config->batch_max_insert_job);
					return -1;
				}

				mega_job_insert_t *job = &(buf->insert_job_list[buf->num_insert_job]);
				job->it = it;

				ielem_t *in = insert_buf->insert_in;
				(in[insert_buf->num_insert_job]).hash = hash; /* the higher 32 bits */
				(in[insert_buf->num_insert_job]).sig = sig1; /* the lower 32 bits */
				(in[insert_buf->num_insert_job]).loc = it->loc;

				/* Update batch parameters */
			#if !defined(RECEIVER_PERFORMANCE_TEST)
				insert_buf->num_insert_job ++;
				buf->num_insert_job ++;
				assert(insert_buf->num_insert_job <= config->batch_max_insert_job);
				assert(buf->num_insert_job <= config->batch_max_insert_job * INSERT_BLOCK);
			#endif

				if (id != batch->receiver_buf_id) {
					/* It will not go wrong if the buffer switches, if the buffer is switched before 
					 * num_insert_job ++, then the job will not be inserted into GPU hash table,
					 * else it will be successfully inserted. Sender will not process the inserted
					 * job. If the item is allocated, but not inserted into the hash_table, we expect
					 * the evict process will handle this.
					 * There may be situations that Sender sets num_insert_job to 0 before --,
					 * we make a test after the operation to guarantee the correctness */
					buf->num_insert_job --;
					if (buf->num_insert_job < 0) buf->num_insert_job = 0;
					insert_buf->num_insert_job --;
					if (insert_buf->num_insert_job < 0) insert_buf->num_insert_job = 0;
					goto reinsert;
				}
			} while(0);

			ptr += (nkey + nbyte);
			break;

		default:
			mprint(ERROR, "Wrong job type\n");
			return -1;
		}

		if (ptr >= payload_ptr + payload_len) {
			mprint(ERROR, "Exceed payload length\n");
			return -1;
		}

	} /* for loop */

	mprint(ERROR, "%d jobs in this packet\n", kk);
	return 0;
}

#if defined(PRELOAD)
static void preload(int id)
{
	/*eth&ip&udp header len + magic number + end mark*/
	int reserved_len = config->eiu_hdr_len + MEGA_MAGIC_NUM_LEN + 2;

	char *payload_ptr, *ptr;
	uint32_t payload_len;

	char *pkt_ptr = malloc(ETHERNET_MAX_FRAME_LEN);

	struct ether_hdr *ethh = (struct ether_hdr *)pkt_ptr;
	struct ipv4_hdr *iph = (struct ipv4_hdr *)(ethh + 1);
	struct udp_hdr *udph = (struct udp_hdr *)(iph + 1);
	payload_ptr = (char *)(udph + 1); // UDP header length: 8 bytes

	*(uint16_t *)payload_ptr = PROTOCOL_MAGIC;
	payload_ptr += MEGA_MAGIC_NUM_LEN;

	const uint32_t preload_cnt = (uint32_t)(LOAD_FACTOR * ((1 << MEM_P)/8));

	uint32_t set_key = 1;


	/* only one thread preload the hash table */
	assert (id == 0);

	mprint(INFO, "[PRELOADING...]: Going to insert %u keys\n", preload_cnt);
	/* preload the keys */
	while (set_key < preload_cnt) {
		ptr = payload_ptr; /* rewrite a packet */
		payload_len = reserved_len; /* basic length */
		iph->src_addr = iph->src_addr + 1; /* ip + 1 for each packet */

		/* construct a packet */
		/* ----------------------------------------------------- */
		while (payload_len + SET_LEN <= ETHERNET_MAX_FRAME_LEN) {
			*(uint16_t *)ptr = MEGA_JOB_SET;
			ptr += sizeof(uint16_t);
			*(uint16_t *)ptr = KEY_LEN;
			ptr += sizeof(uint16_t);
			*(uint32_t *)ptr = VALUE_LEN;
			ptr += sizeof(uint32_t);

			if (config->bits_insert_buf == 0)
				*(uint32_t *)(ptr + sizeof(uint32_t)) = set_key;
			else
				*(uint32_t *)(ptr + sizeof(uint32_t)) = (rte_bswap32(set_key & 0xff) << (8 - config->bits_insert_buf)) | (set_key);
			*(uint32_t *)(ptr) = set_key;

			ptr += KEY_LEN;
			ptr += VALUE_LEN;

			payload_len += SET_LEN;

			set_key ++;
			if (set_key >= preload_cnt) {
				break;
			}
		}

		/* write the ending mark */
		*(uint16_t *)ptr = 0xFFFF;

		process_packet(pkt_ptr, payload_len, id);

		/* reduce insert speed */
		int k = 10000;
		while(k > 0) k--;
	}

	mprint(INFO, " ==========================     Hash table has been loaded     ========================== \n");

	loading_mode = 0;

	free(pkt_ptr);

	return ;
}
#endif

#if defined(LOCAL_TEST)
static void mega_receiver_local_read(int id)
{
	mega_receiver_t *cc = &(receivers[id]);

	/*eth&ip&udp header len + magic number + end mark*/
	int reserved_len = config->eiu_hdr_len + MEGA_MAGIC_NUM_LEN + 2;

	gettimeofday(&(cc->startime), NULL);

	char *payload_ptr, *ptr;
	uint32_t payload_len;

	int num_get = NUM_DEFINED_GET;
	int num_set = NUM_DEFINED_SET;

	char *pkt_ptr = malloc(ETHERNET_MAX_FRAME_LEN);

	struct ether_hdr *ethh = (struct ether_hdr *)pkt_ptr;
	struct ipv4_hdr *iph = (struct ipv4_hdr *)(ethh + 1);
	struct udp_hdr *udph = (struct udp_hdr *)(iph + 1);
	payload_ptr = (char *)(udph + 1); // UDP header length: 8 bytes

	*(uint16_t *)payload_ptr = PROTOCOL_MAGIC;
	payload_ptr += MEGA_MAGIC_NUM_LEN;

	const uint32_t preload_cnt = (uint32_t)(LOAD_FACTOR * ((1 << MEM_P)/8));
	const uint32_t total_cnt = ((uint32_t)1 << 31) - 1;

	struct zipf_gen_state state;
	mehcached_zipf_init(&state, (uint64_t)preload_cnt - 2, (double)ZIPF_THETA, (uint64_t)21);

#if defined(PRELOAD)
	while (loading_mode == 1) ;
#endif

	/* Different receivers use different keys start point */
	uint32_t get_key = (10000 * id) % preload_cnt;
	uint32_t set_key = preload_cnt + id * ((total_cnt - preload_cnt)/config->cpu_worker_num);

	for (;;) {
		ptr = payload_ptr; /* rewrite a packet */
		payload_len = reserved_len; /* basic length */
		iph->src_addr = iph->src_addr + 1; /* ip + 1 for each packet */

		/* construct a packet */
		/* ----------------------------------------------------- */
		while (1) {
			if (num_get > 0) {
				if (payload_len + GET_LEN > ETHERNET_MAX_FRAME_LEN) {
					break;
				}
				*(uint16_t *)ptr = MEGA_JOB_GET;
				/* skip job_type, key length = 4 bytes in total */
				ptr += sizeof(uint16_t);
				*(uint16_t *)ptr = KEY_LEN;
				ptr += sizeof(uint16_t);

				get_key = (uint32_t)mehcached_zipf_next(&state) + 1;
				assert (get_key < preload_cnt);

				/* here we try to evenly distribute the key through insert bufs,
				 * on the first 32 bits, the highest 5 bits are used for 32 insert bufs, 
				 * rte_bswap32(key & 0xff) << 3 is to assign the 5 bits.
				 * We also need to distribute keys among buckets, and it is the lower
				 * bits are used for hash. the "|key" is setting the hash.
				 * The next 32 bits are used as signature, just key ++ */
				if (config->bits_insert_buf == 0)
					*(uint32_t *)(ptr + sizeof(uint32_t)) = get_key;
				else
					*(uint32_t *)(ptr + sizeof(uint32_t)) = (rte_bswap32(get_key & 0xff) << (8 - config->bits_insert_buf)) | get_key;
				*(uint32_t *)(ptr) = get_key;

				ptr += KEY_LEN;

				payload_len += GET_LEN;

				num_get --;
				if (num_get == 0 && NUM_DEFINED_SET > 0) {
					num_set = NUM_DEFINED_SET;
				} else if (num_get == 0) {
					num_get = NUM_DEFINED_GET;
				}
			} else if (num_set > 0){
				if (payload_len + SET_LEN > ETHERNET_MAX_FRAME_LEN) {
					break;
				}
				*(uint16_t *)ptr = MEGA_JOB_SET;
				ptr += sizeof(uint16_t);
				*(uint16_t *)ptr = KEY_LEN;
				ptr += sizeof(uint16_t);
				*(uint32_t *)ptr = VALUE_LEN;
				ptr += sizeof(uint32_t);

				set_key ++;
				if (set_key >= preload_cnt + (id + 1) * ((total_cnt - preload_cnt)/config->cpu_worker_num)) {
					/* TODO: longer test */
					exit(0);
				}

				if (config->bits_insert_buf == 0)
					*(uint32_t *)(ptr + sizeof(uint32_t)) = set_key;
				else
					*(uint32_t *)(ptr + sizeof(uint32_t)) = (rte_bswap32(set_key & 0xff) << (8 - config->bits_insert_buf)) | (set_key);
				*(uint32_t *)(ptr) = set_key;

				ptr += KEY_LEN;
				ptr += VALUE_LEN;

				payload_len += SET_LEN;

				num_set --;
				if (num_set == 0 && NUM_DEFINED_GET > 0) {
					num_get = NUM_DEFINED_GET;
				} else if (num_set == 0) {
					num_set = NUM_DEFINED_SET;
				}
			} else {
				mprint(ERROR, "ERROR\n");
				exit(0);
			}

			/* a loop to reduce generation speed to get low cycle latency */
			//int k = 10;
			//while(k > 0) k--;
		}

		/* write the ending mark */
		*(uint16_t *)ptr = 0xFFFF;

		cc->total_packets ++;
		cc->total_bytes += (uint64_t)payload_len;

		process_packet(pkt_ptr, payload_len, id);
	}
}

#else /* LOCAL_TEST */
static void mega_receiver_read(int ifindex, int queue_id, int id)
{
	int ret, i;
	struct rte_mbuf *m, **pkts_burst;

	pkts_burst = (struct rte_mbuf **)malloc(config->io_batch_num * sizeof(struct rte_mbuf *));

	mega_receiver_t *cc = &(receivers[id]);

	gettimeofday(&(cc->startime), NULL);

	for (;;) {

		ret = rte_eth_rx_burst((uint8_t)ifindex, queue_id, pkts_burst, config->io_batch_num);
		cc->total_packets += ret;

		for (i = 0; i < ret; i ++) {
			m = pkts_burst[i];
			rte_prefetch0(rte_pktmbuf_mtod(m, void *));
			rte_prefetch0(64 + (char *)rte_pktmbuf_mtod(m, void *));
			/*
			if (m->pkt.pkt_len < 1300 || m->pkt.nb_segs != 1) {
				mprint(ERROR, "wrong packet : %d\n", m->pkt.pkt_len);
				break;
			}
			*/
			cc->total_bytes += (uint64_t)m->pkt.pkt_len;

#if !defined(NOT_COLLECT)
			process_packet(m->pkt.data, m->pkt.pkt_len, id);
#endif
			rte_pktmbuf_free(pkts_burst[i]);
		}
	}
}
#endif /* LOCAL_TEST */

void *mega_receiver_main(mega_receiver_context_t *context)
{
	mega_receiver_init(context);
	mprint(INFO, "[Receiver %d] on core %d is attaching if:queue %d:%d ...\n", 
			context->unique_id, context->core_id, context->ifindex, context->queue_id);
	fflush(stdout);

#if defined(PRELOAD)
	if (context->unique_id == 0) {
		preload(context->unique_id);
	}
#endif

#if defined(LOCAL_TEST)
	mega_receiver_local_read(context->unique_id);
#else
	mega_receiver_read(context->ifindex, context->queue_id, context->unique_id);
#endif

	exit(0);
}
