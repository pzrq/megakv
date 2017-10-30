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
#include <sys/time.h>
#include <string.h>
#include <sched.h>
#include <assert.h>

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
#include <rte_udp.h>
#include <rte_byteorder.h>
#include <rte_ethdev.h>
#include <rte_ring.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>

#include "mega_context.h"
#include "mega_sender.h"
#include "mega_config.h"
#include "mega_memory.h"
#include "mega_macros.h"
#include "mega_job.h"
#include "mega_batch.h"
#include "mega_common.h"
#include "slabs.h"
#include "macros.h"

extern pthread_key_t worker_batch_struct;
extern mega_config_t *config;
extern pthread_mutex_t mutex_worker_init;
extern slabclass_t slabclass[MAX_NUMBER_OF_SLAB_CLASSES];
extern struct rte_mempool * send_pktmbuf_pool;

mega_sender_t senders[MAX_WORKER_NUM];

struct mbuf_table {
	unsigned len;
	struct rte_mbuf **m_table;
};

static int mega_sender_init(mega_sender_context_t *context)
{
	unsigned int i;
	mega_batch_t *batch = context->batch;

	for (i = 0; i < config->cpu_worker_num; i ++) {
		senders[i].total_packets = 0;
		senders[i].total_bytes = 0;
		senders[i].hit = 0;
		senders[i].miss = 0;
	}

#if defined(CPU_AFFINITY)
	/* set schedule affinity */
	unsigned long mask = 1 << context->core_id;
	if (sched_setaffinity(0, sizeof(unsigned long), (cpu_set_t *)&mask) < 0) {
		mprint(ERROR, "set affinity on core %d\n", context->core_id);
		assert(0);
	}

#if 0
	/* set schedule policy */
	struct sched_param param;
	param.sched_priority = 98;
	pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
#endif
#endif

	batch->sender_buf_id = -1;
	
	pthread_setspecific(worker_batch_struct, (void *)batch);
	__builtin_prefetch(batch);
	__builtin_prefetch(&worker_batch_struct);

	pthread_mutex_lock(&mutex_worker_init);
	context->initialized = 1;
	pthread_mutex_unlock(&mutex_worker_init);

	return 0;
}

static inline int mega_sender_refresh_buffer(mega_batch_buf_t *buf, mega_batch_t *batch)
{
	/* refresh receiver_buf  */
	buf->num_compact_search_job = 0;
	buf->num_search_job = 0;
	if (batch->delay == 0) {
		buf->num_insert_job = 0;
		buf->num_delete_job = 0;

		unsigned int i;
		/* clear all the insert bufs, FIXME: clear once is enough,
		 * here we cleared for worker_num times, because this is shared */
		for (i = 0; i < config->num_insert_buf; i ++) {
			buf->insert_buf[i].num_insert_job = 0;
		}
	}

	return 0;
}

static int mega_sender_give_available_buffer(int id)
{
	mega_batch_t *batch = pthread_getspecific(worker_batch_struct);
	mega_batch_buf_t *buf = &(batch->buf[batch->sender_buf_id]);
	
	/* Make the buffer looks like new */
	mega_sender_refresh_buffer(buf, batch);

	mprint(DEBUG, "<<< [sender %d] < give available buffer %d\n", id, batch->sender_buf_id);
	/* tell the receiver that the buffer is available */
#if defined(USE_LOCK)
	pthread_mutex_lock(&(batch->mutex_available_buf_id));
	if (batch->available_buf_id[0] == -1) {
		batch->available_buf_id[0] = batch->sender_buf_id;
	} else if (batch->available_buf_id[1] == -1) {
		batch->available_buf_id[1] = batch->sender_buf_id;
	} else {
		mprint(ERROR, "Three buffers available\n");
		assert(0);
	}
	pthread_mutex_unlock(&(batch->mutex_available_buf_id));

	batch->sender_buf_id = -1;
#else
	if (batch->available_buf_id[0] == -1) {
		batch->available_buf_id[0] = batch->sender_buf_id;
	} else if (batch->available_buf_id[1] == -1) {
		batch->available_buf_id[1] = batch->sender_buf_id;
	} else {
		mprint(ERROR, "Three buffers available\n");
		assert(0);
	}

	batch->sender_buf_id = -1;
#endif

	return 0;
}

static int mega_sender_get_buffer(void)
{
	mega_batch_t *batch = pthread_getspecific(worker_batch_struct);

	/* wait for the gpu worker to give me the buffer ~~ */
	while(batch->sender_buf_id == -1) {
		;
	}

	return 0;
}

#if defined(KEY_MATCH)
#define WRITE_PKT(pkt_ptr, item) \
		do { \
			*(uint16_t *)pkt_ptr = PROTOCOL_VALUE; \
			pkt_ptr += sizeof(uint16_t); \
			*(uint32_t *)pkt_ptr = item->nbytes; \
			pkt_ptr += sizeof(item->nbytes); \
			rte_memcpy(pkt_ptr, ITEM_value(item), item->nbytes); \
			pkt_ptr += item->nbytes; \
		} while(0)
#else
#define WRITE_PKT(pkt_ptr, item) \
		do { \
			*(uint16_t *)pkt_ptr = PROTOCOL_VALUE; \
			pkt_ptr += sizeof(uint16_t); \
			*(uint32_t *)pkt_ptr = item->nbytes; \
			pkt_ptr += sizeof(item->nbytes); \
			rte_memcpy(pkt_ptr, ITEM_key(item), item->nbytes + item->nkey); \
			pkt_ptr += item->nbytes; \
		} while(0)
#endif

#define ITEM_GET(it, loc) \
		do { \
			slabclass_t *p = &(slabclass[loc >> (config->loc_bits - SLAB_ID_BITS)]); \
			unsigned int offset = (loc << SLAB_ID_BITS) >> SLAB_ID_BITS; \
			assert (offset < p->perslab * p->slabs); \
			it = (item *)((char *)(p->slab_list[offset >> p->perslab_bits]) + p->size * (offset & ((1 << p->perslab_bits) - 1))); \
			(p->bitmap).map[offset >> BM_MASK_BIT] &= ~((uint64_t)1 << (offset & ((1 << BM_MASK_BIT) - 1))); \
		} while(0)

#define PREFETCH_ITEM(loc) \
		do { \
			if (loc != 0) { \
				slabclass_t *p = &(slabclass[loc >> (config->loc_bits - SLAB_ID_BITS)]); \
				unsigned int offset = (loc << SLAB_ID_BITS) >> SLAB_ID_BITS; \
				item *it = (item *)((char *)(p->slab_list[offset >> p->perslab_bits]) + p->size * (offset & ((1 << p->perslab_bits) - 1))); \
				__builtin_prefetch(it, 0, 0); \
			} \
		} while(0)

/* This function trigger write event of all the jobs in this batch */
static int mega_sender_sendloop(int ifindex, int queue_id, int id)
{
	int pkt_mark, job_mark, total_cnt;
	uint32_t loc, i;
	uint32_t index, total_length;

	struct ether_hdr *ethh;
	struct ipv4_hdr *iph;
	struct udp_hdr *udph;
	item *it;
	char *pkt_ptr, *pkt_base_ptr;

	mega_job_search_t *this_job;
	mega_batch_t *batch = pthread_getspecific(worker_batch_struct);
	mega_batch_buf_t *buf;

	/* Initialize ioengine */
	mega_sender_t *cc = &(senders[id]);

#if defined(LOCAL_TEST)
	char *packets = (char *)malloc(ETHERNET_MAX_FRAME_LEN * config->io_batch_num);
	char *m;
	for (i = 0; i < config->io_batch_num; i ++) {
		m = packets + i * ETHERNET_MAX_FRAME_LEN;
		assert (m != NULL);

		ethh = (struct ether_hdr *)m;
		ethh->s_addr = config->local_mac_addr;
		ethh->ether_type = rte_cpu_to_be_16((uint16_t)(ETHER_TYPE_IPv4));

		iph = (struct ipv4_hdr *)((unsigned char *)ethh + sizeof(struct ether_hdr));
		iph->version_ihl = 0x40 | 0x05;
		iph->type_of_service = 0;
		iph->packet_id = 0;
		iph->fragment_offset = 0;
		iph->time_to_live = 64;
		iph->next_proto_id = IPPROTO_UDP;
		iph->hdr_checksum = 0;
		iph->src_addr = config->local_ip_addr;

		udph = (struct udp_hdr *)((unsigned char *)iph + sizeof(struct ipv4_hdr));
		udph->src_port = config->local_udp_port;
		udph->dgram_cksum = 0;
	}
#else
	int ret;
	struct mbuf_table tx_mbufs;
	struct rte_mbuf *m;

	tx_mbufs.m_table = (struct rte_mbuf **)malloc(sizeof(struct rte_mbuf *) * config->io_batch_num);
	for (i = 0; i < config->io_batch_num; i ++) {
		m = rte_pktmbuf_alloc(send_pktmbuf_pool);
		assert (m != NULL);
		m->pkt.nb_segs = 1;
		m->pkt.next = NULL;
		tx_mbufs.m_table[i] = m;

		ethh = (struct ether_hdr *)rte_pktmbuf_mtod(m, unsigned char *);
		ethh->s_addr = config->local_mac_addr;
		ethh->ether_type = rte_cpu_to_be_16((uint16_t)(ETHER_TYPE_IPv4));

		iph = (struct ipv4_hdr *)((unsigned char *)ethh + sizeof(struct ether_hdr));
		iph->version_ihl = 0x40 | 0x05;
		iph->type_of_service = 0;
		iph->packet_id = 0;
		iph->fragment_offset = 0;
		iph->time_to_live = 64;
		iph->next_proto_id = IPPROTO_UDP;
		iph->hdr_checksum = 0;
		iph->src_addr = config->local_ip_addr;

		udph = (struct udp_hdr *)((unsigned char *)iph + sizeof(struct ipv4_hdr));
		udph->src_port = config->local_udp_port;
		udph->dgram_cksum = 0;
	}
#endif

	gettimeofday(&(cc->startime), NULL);

	for (;;) {
		/* Get buffer first */
		mega_sender_get_buffer();

		buf = &(batch->buf[batch->sender_buf_id]);

#if defined(NOT_FORWARD)
#if 0
		if (batch->delay == 0) {
			for (i = 0; i < buf->num_insert_job; i ++) {
				mega_job_insert_t *job = &(buf->insert_job_list[i]); 
				/* Here we do not care about the results of insert. Just set the flags from
				 * ITEM_WRITING to 0. Because we believe that the chance for conflicting
				 * signature is really small, and the evicting process will handle it.
				 * The primary reason that we make this simple is to avoid concurrent 
				 * operations with the receiver, avoiding locks and sophisticated logic
				 * on the critical code path */
				if (job->it->flags != ITEM_WRITING) {
					mprint(ERROR, "[sender] flag is not ITEM_WRITING for insert job, i = %d\n", i);
				}
				job->it->flags = 0;
			}
		}
#endif
		mega_sender_give_available_buffer(id);
		continue;
#endif

		//if (buf->num_search_job + buf->num_insert_job == 0) {
		if (buf->num_search_job == 0) {
			mprint(DEBUG, "[sender %d] gets 0 packets to forward\n", id);
			mega_sender_give_available_buffer(id);
			continue;
		}


#if defined(COMPACT_JOB)
		job_mark = 1;
		total_cnt = buf->num_compact_search_job + 1;
#else
		job_mark = 0;
		total_cnt = buf->num_compact_search_job;
#endif

/* =================================================================================================*/
#if defined(PREFETCH_BATCH)
		pkt_mark = 0;
	#if (KVSIZE > 1)
		const int prefetch_cache_cnt = 1 + (MAX_KEY_LENGTH + MAX_VALUE_LENGTH + 64) / 64; 
		int p_cnt;
	#endif

		this_job = &(buf->search_job_list[job_mark]);

	#if defined(LOCAL_TEST)
		ethh = (struct ether_hdr *)(&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]));
		ethh->d_addr = this_job->eth_addr;
		iph = (struct ipv4_hdr *)((struct ether_hdr *)(&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark])) + 1);
	#else
		ethh = (struct ether_hdr *)((tx_mbufs.m_table[pkt_mark])->pkt.data);
		ethh->d_addr = this_job->eth_addr;
		iph = (struct ipv4_hdr *)((struct ether_hdr *)((tx_mbufs.m_table[pkt_mark])->pkt.data) + 1);
	#endif
		iph->dst_addr = this_job->ip;
		udph = (struct udp_hdr *)(iph + 1);
		udph->dst_port = this_job->port;
	#if defined(LOCAL_TEST)
		pkt_ptr = (char *)&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]) + config->eiu_hdr_len;
		pkt_base_ptr = &(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]);
	#else
		pkt_ptr = (char *)((tx_mbufs.m_table[pkt_mark])->pkt.data) + config->eiu_hdr_len;
		pkt_base_ptr = (tx_mbufs.m_table[pkt_mark])->pkt.data;
	#endif

		index = this_job->index;
		int to_send = 0;
		slabclass_t *p;
		unsigned int offset;
		int i, k;

		item *item_pointer_buf[PREFETCH_BATCH_DISTANCE];
	#if defined(KEY_MATCH)
		char *key_buf[PREFETCH_BATCH_DISTANCE];
		int key_len_buf[PREFETCH_BATCH_DISTANCE];
		char *key;
		int key_len;
	#endif

		while (index < buf->num_search_job) {
			i = 0;
prefetch_loop1:
			loc = buf->search_out[index << 1];
			if (loc == 0) {
				loc = buf->search_out[(index << 1) + 1];
			}

			/* find the corresponding job */
			if (this_job->num > 0) {
				if (index != this_job->index) {
					/* maybe something wrong with the buffer switch on the last job */
					to_send = 1;
					index ++;
					goto break_loop1;
				}
				this_job->num --;
				this_job->index ++;
				index ++;
			} else {
				job_mark ++;
				if (job_mark >= total_cnt) {
					to_send = 1;
					goto break_loop1;
				}
				this_job = &(buf->search_job_list[job_mark]);
				goto prefetch_loop1;
			}


			if (loc != 0) {
				p = &(slabclass[loc >> (config->loc_bits - SLAB_ID_BITS)]);
				offset = (loc << SLAB_ID_BITS) >> SLAB_ID_BITS;
				it = (item *)((char *)(p->slab_list[offset >> p->perslab_bits]) + p->size * (offset & ((1 << p->perslab_bits) - 1)));
				item_pointer_buf[i] = it;
		#if defined(KEY_MATCH)
				key_buf[i] = this_job->key;
				key_len_buf[i] = this_job->nkey;
		#endif

				/* NOTE: Even for 8B key/value item, it may span two cache lines,
				 * therefore, prefetch the next cache line would further improve  performance.
				 * TODO: why using direct prefetch instead of following loops would bring more performance?
				 * For KVSIZE 0 and 1, up to 10% performance differences are observed */
			#if (KVSIZE == 0)
				__builtin_prefetch(it, 0, 0);
				__builtin_prefetch((char *)it + 64, 0, 0);
			#elif (KVSIZE == 1)
				__builtin_prefetch(it, 0, 0);
				__builtin_prefetch((char *)it + 64, 0, 0);
				__builtin_prefetch((char *)it + 128, 0, 0);
			#else
				for (p_cnt = 0; p_cnt < prefetch_cache_cnt; p_cnt ++) {
					__builtin_prefetch((char *)it + 64 * p_cnt, 0, 0);
				}
			#endif
			} else {
				cc->miss ++;
				goto prefetch_loop1;
			}

			i ++;
			if (i < PREFETCH_BATCH_DISTANCE)	goto prefetch_loop1;

break_loop1:
			if (!(i >=1 && i <= PREFETCH_BATCH_DISTANCE)) {
				break;
			}
			k = 0;
prefetch_loop2:
			it = item_pointer_buf[k];
		#if defined(KEY_MATCH)
			key = key_buf[k]; 
			key_len = key_len_buf[k];
		#endif

			if (it != NULL) {
				/* if this writing this job will exceed the maximum frame length or the ip/port does
					not belong to this packet, complete this packet and use a new one*/
		#if defined(KEY_MATCH)
				if (pkt_ptr - pkt_base_ptr + it->nbytes + 6 > ETHERNET_MAX_FRAME_LEN ||
					this_job->ip != iph->dst_addr || this_job->port != udph->dst_port) { 
		#else
				if (pkt_ptr - pkt_base_ptr + it->nkey + it->nbytes + 6 > ETHERNET_MAX_FRAME_LEN || 
					this_job->ip != iph->dst_addr || this_job->port != udph->dst_port) { 
		#endif

					/* complete this packet */
					total_length = pkt_ptr - pkt_base_ptr;
				#if !defined(LOCAL_TEST)
					(tx_mbufs.m_table[pkt_mark])->pkt.pkt_len
						= (tx_mbufs.m_table[pkt_mark])->pkt.data_len
						= total_length;
				#endif
					iph->total_length = rte_cpu_to_be_16((uint16_t)(total_length - sizeof(struct ether_hdr))); 
					udph->dgram_len = rte_cpu_to_be_16((uint16_t)(total_length - sizeof(struct ether_hdr) - sizeof(struct ipv4_hdr)));
					cc->total_bytes += total_length;

					/* if we should send the chunk */
					if (pkt_mark == config->io_batch_num - 1) {
				#if defined(LOCAL_TEST)
						cc->total_packets += config->io_batch_num;
				#else
						tx_mbufs.len = pkt_mark;
						ret = rte_eth_tx_burst(ifindex, (uint16_t)queue_id, tx_mbufs.m_table, (uint16_t)tx_mbufs.len);
						cc->total_packets += ret;
						cc->dropped_packets += tx_mbufs.len - ret;
				#endif
						pkt_mark = 0;
					} else {
						pkt_mark ++;
					}

					/* new packet */
				#if defined(LOCAL_TEST)
					ethh = (struct ether_hdr *)(&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]));
					ethh->d_addr = this_job->eth_addr;
					iph = (struct ipv4_hdr *)((struct ether_hdr *)(&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark])) + 1);
				#else
					ethh = (struct ether_hdr *)((tx_mbufs.m_table[pkt_mark])->pkt.data);
					ethh->d_addr = this_job->eth_addr;
					iph = (struct ipv4_hdr *)((struct ether_hdr *)((tx_mbufs.m_table[pkt_mark])->pkt.data) + 1);
				#endif
					iph->dst_addr = this_job->ip;
					udph = (struct udp_hdr *)(iph + 1);
					udph->dst_port = this_job->port;
				#if defined(LOCAL_TEST)
					pkt_ptr = (char *)&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]) + config->eiu_hdr_len;
					pkt_base_ptr = &(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]);
				#else
					pkt_ptr = (char *)((tx_mbufs.m_table[pkt_mark])->pkt.data) + config->eiu_hdr_len;
					pkt_base_ptr = (tx_mbufs.m_table[pkt_mark])->pkt.data;
				#endif
					__builtin_prefetch(pkt_ptr, 0, 0);
				}
		#if defined(KEY_MATCH)
				if ((it->nkey == key_len) && (strncmp(ITEM_key(it), key, key_len) == 0)) {
					WRITE_PKT(pkt_ptr, it);
					cc->hit ++;
				} else {
					cc->miss ++;
				}
		#else
				WRITE_PKT(pkt_ptr, it);
				cc->hit ++;
		#endif	
			}
		#if !defined(KEY_MATCH)
			else {
				cc->miss ++;
			}
		#endif

			k ++;
			if (k < i) {
				goto prefetch_loop2;
			}
			if (i != PREFETCH_BATCH_DISTANCE) {
				break;
			}
		}

		if (to_send == 1) {
			/* complete this packet */
			total_length = pkt_ptr - pkt_base_ptr;
		#if !defined(LOCAL_TEST)
			(tx_mbufs.m_table[pkt_mark])->pkt.pkt_len
				= (tx_mbufs.m_table[pkt_mark])->pkt.data_len
				= total_length;
		#endif
			iph->total_length = rte_cpu_to_be_16((uint16_t)(total_length - sizeof(struct ether_hdr))); 
			udph->dgram_len = rte_cpu_to_be_16((uint16_t)(total_length - sizeof(struct ether_hdr) - sizeof(struct ipv4_hdr)));
			cc->total_bytes += total_length;

			/* send the chunk */
		#if defined(LOCAL_TEST)
			cc->total_packets += config->io_batch_num;
		#else
			tx_mbufs.len = pkt_mark + 1;
			ret = rte_eth_tx_burst(ifindex, (uint16_t)queue_id, tx_mbufs.m_table, (uint16_t)tx_mbufs.len);
			cc->total_packets += ret;
			cc->dropped_packets += tx_mbufs.len - ret;
		#endif
		}

/* =================================================================================================*/
#elif defined(PREFETCH_BATCH_NO)
		pkt_mark = 0;

		this_job = &(buf->search_job_list[job_mark]);

	#if defined(LOCAL_TEST)
		ethh = (struct ether_hdr *)(&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]));
		ethh->d_addr = this_job->eth_addr;
		iph = (struct ipv4_hdr *)((struct ether_hdr *)(&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark])) + 1);
	#else
		ethh = (struct ether_hdr *)((tx_mbufs.m_table[pkt_mark])->pkt.data);
		ethh->d_addr = this_job->eth_addr;
		iph = (struct ipv4_hdr *)((struct ether_hdr *)((tx_mbufs.m_table[pkt_mark])->pkt.data) + 1);
	#endif
		iph->dst_addr = this_job->ip;
		udph = (struct udp_hdr *)(iph + 1);
		udph->dst_port = this_job->port;
	#if defined(LOCAL_TEST)
		pkt_ptr = (char *)&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]) + config->eiu_hdr_len;
		pkt_base_ptr = &(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]);
	#else
		pkt_ptr = (char *)((tx_mbufs.m_table[pkt_mark])->pkt.data) + config->eiu_hdr_len;
		pkt_base_ptr = (tx_mbufs.m_table[pkt_mark])->pkt.data;
	#endif

		index = this_job->index;
		int to_send = 0;

		while (index < buf->num_search_job) {
			/* find the corresponding job */
			if (this_job->num > 0) {
				if (index != this_job->index) {
					/* maybe something wrong with the buffer swith on the last job */
					to_send = 1;
					break;
				}
				this_job->num --;
				this_job->index ++;
			} else {
				job_mark ++;
				if (job_mark >= total_cnt) {
					to_send = 1;
					break;
				}
				this_job = &(buf->search_job_list[job_mark]);
				continue;
			}

			if (buf->search_out[index << 1] != 0) {
				loc = buf->search_out[index << 1];
			} else if (buf->search_out[(index << 1) + 1] != 0) {
				loc = buf->search_out[(index << 1) + 1];
			} else {
				continue;
			}

			index ++;

			ITEM_GET(it, loc);
			if (it != NULL) {
				/* if this writing this job will exceed the maximum frame length or the ip/port does
					not belong to this packet, complete this packet and use a new one*/
		#if defined(KEY_MATCH)
				if (pkt_ptr - pkt_base_ptr + it->nbytes + 6 > ETHERNET_MAX_FRAME_LEN ||
					this_job->ip != iph->dst_addr || this_job->port != udph->dst_port) { 
		#else
				if (pkt_ptr - pkt_base_ptr + it->nkey + it->nbytes + 6 > ETHERNET_MAX_FRAME_LEN || 
					this_job->ip != iph->dst_addr || this_job->port != udph->dst_port) { 
		#endif

					/* complete this packet */
					total_length = pkt_ptr - pkt_base_ptr;
				#if !defined(LOCAL_TEST)
					(tx_mbufs.m_table[pkt_mark])->pkt.pkt_len
						= (tx_mbufs.m_table[pkt_mark])->pkt.data_len
						= total_length;
				#endif
					iph->total_length = rte_cpu_to_be_16((uint16_t)(total_length - sizeof(struct ether_hdr))); 
					udph->dgram_len = rte_cpu_to_be_16((uint16_t)(total_length - sizeof(struct ether_hdr) - sizeof(struct ipv4_hdr)));
					cc->total_bytes += total_length;

					/* if we should send the chunk */
					if (pkt_mark == config->io_batch_num - 1) {
				#if defined(LOCAL_TEST)
						cc->total_packets += config->io_batch_num;
				#else
						tx_mbufs.len = pkt_mark + 1;
						ret = rte_eth_tx_burst(ifindex, (uint16_t)queue_id, tx_mbufs.m_table, (uint16_t)tx_mbufs.len);
						cc->total_packets += ret;
						cc->dropped_packets += tx_mbufs.len - ret;
				#endif
						pkt_mark = 0;
					} else {
						pkt_mark ++;
					}

					/* new packet */
				#if defined(LOCAL_TEST)
					ethh = (struct ether_hdr *)(&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]));
					ethh->d_addr = this_job->eth_addr;
					iph = (struct ipv4_hdr *)((struct ether_hdr *)(&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark])) + 1);
				#else
					ethh = (struct ether_hdr *)((tx_mbufs.m_table[pkt_mark])->pkt.data);
					ethh->d_addr = this_job->eth_addr;
					iph = (struct ipv4_hdr *)((struct ether_hdr *)((tx_mbufs.m_table[pkt_mark])->pkt.data) + 1);
				#endif
					iph->dst_addr = this_job->ip;
					udph = (struct udp_hdr *)(iph + 1);
					udph->dst_port = this_job->port;
				#if defined(LOCAL_TEST)
					pkt_ptr = (char *)&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]) + config->eiu_hdr_len;
					pkt_base_ptr = &(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]);
				#else
					pkt_ptr = (char *)((tx_mbufs.m_table[pkt_mark])->pkt.data) + config->eiu_hdr_len;
					pkt_base_ptr = (tx_mbufs.m_table[pkt_mark])->pkt.data;
				#endif
				}
		#if defined(KEY_MATCH)
				if ((it->nkey == this_job->nkey) && (strncmp(ITEM_key(it), this_job->key, it->nkey) == 0)) {
					WRITE_PKT(pkt_ptr, it);
					cc->hit ++;
				} else {
					cc->miss ++;
				}
		#else
				WRITE_PKT(pkt_ptr, it);
				cc->hit ++;
		#endif	
			}
		#if !defined(KEY_MATCH)
			else {
				cc->miss ++;
			}
		#endif
		}

		if (to_send == 1) {
			/* complete this packet */
			total_length = pkt_ptr - pkt_base_ptr;
		#if !defined(LOCAL_TEST)
			(tx_mbufs.m_table[pkt_mark])->pkt.pkt_len
				= (tx_mbufs.m_table[pkt_mark])->pkt.data_len
				= total_length;
		#endif
			iph->total_length = rte_cpu_to_be_16((uint16_t)(total_length - sizeof(struct ether_hdr))); 
			udph->dgram_len = rte_cpu_to_be_16((uint16_t)(total_length - sizeof(struct ether_hdr) - sizeof(struct ipv4_hdr)));
			cc->total_bytes += total_length;

			/* send the chunk */
		#if defined(LOCAL_TEST)
			cc->total_packets += config->io_batch_num;
		#else
			tx_mbufs.len = pkt_mark + 1;
			ret = rte_eth_tx_burst(ifindex, (uint16_t)queue_id, tx_mbufs.m_table, (uint16_t)tx_mbufs.len);
			cc->total_packets += ret;
			cc->dropped_packets += tx_mbufs.len - ret;
		#endif
		}

/* =================================================================================================*/
#else
		while (job_mark < total_cnt) {

			pkt_mark = 0;

			/* construct a chunk */
			while ((pkt_mark < config->io_batch_num) && (job_mark < total_cnt)) {

				this_job = &(buf->search_job_list[job_mark]);
			#if defined(LOCAL_TEST)
				ethh = (struct ether_hdr *)(&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]));
				ethh->d_addr = this_job->eth_addr;

				iph = (struct ipv4_hdr *)((struct ether_hdr *)(&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark])) + 1);
			#else
				ethh = (struct ether_hdr *)((tx_mbufs.m_table[pkt_mark])->pkt.data);
				ethh->d_addr = this_job->eth_addr;
				
				iph = (struct ipv4_hdr *)((struct ether_hdr *)((tx_mbufs.m_table[pkt_mark])->pkt.data) + 1);
			#endif

				iph->dst_addr = this_job->ip;
				udph = (struct udp_hdr *)(iph + 1);
				udph->dst_port = this_job->port;

			#if defined(LOCAL_TEST)
				pkt_ptr = (char *)&(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]) + config->eiu_hdr_len;
				pkt_base_ptr = &(packets[ETHERNET_MAX_FRAME_LEN * pkt_mark]);
			#else
				pkt_ptr = (char *)((tx_mbufs.m_table[pkt_mark])->pkt.data) + config->eiu_hdr_len;
				pkt_base_ptr = (tx_mbufs.m_table[pkt_mark])->pkt.data;
			#endif

				/* FIXME: corner case-> the last job may lead to no content in a new packet */

				/* construct a packet */
				for (;;) {
					if (job_mark >= total_cnt) {
						break;
					}

					this_job = &(buf->search_job_list[job_mark]);
					if (this_job->ip != iph->dst_addr || this_job->port != udph->dst_port) {
						/* This job does not belong to this packet, we use another one */
						break;
					}
					if (this_job->num > 0) {
						index = this_job->index;
					} else {
						job_mark ++;
						continue;
					}

				#if defined(PREFETCH_PIPELINE)
					if (index + PREFETCH_DISTANCE < buf->num_search_job) {
						PREFETCH_ITEM(buf->search_out[(index + PREFETCH_DISTANCE) << 1]);
					}
				#endif
					loc = buf->search_out[index << 1];
					if (loc != 0) {
						ITEM_GET(it, loc);
						if (it != NULL) {
						#if defined(KEY_MATCH)
							if (pkt_ptr - pkt_base_ptr + it->nbytes + 6 > ETHERNET_MAX_FRAME_LEN) { 
								goto exceed_packet_len;
							}
							if ((it->nkey == this_job->nkey) && (strncmp(ITEM_key(it), this_job->key, it->nkey) == 0)) {
								WRITE_PKT(pkt_ptr, it);
								cc->hit ++;
								goto this_job_end;
							} else {
								mprint(DEBUG, "Key does not match!\n");
							}
						#else
							if (pkt_ptr - pkt_base_ptr + it->nkey + it->nbytes + 6 > ETHERNET_MAX_FRAME_LEN) { 
								goto exceed_packet_len;
							}

							WRITE_PKT(pkt_ptr, it);
							cc->hit ++;
							goto this_job_end;
						#endif
						}
					}

					loc = buf->search_out[(index << 1) + 1];
					if (loc != 0) {
						ITEM_GET(it, loc);
						if (it != NULL) {
						#if defined(KEY_MATCH)
							if (pkt_ptr - pkt_base_ptr + it->nbytes + 6 > ETHERNET_MAX_FRAME_LEN) { 
								goto exceed_packet_len;
							}
							if ((it->nkey == this_job->nkey) && (strncmp(ITEM_key(it), this_job->key, it->nkey) == 0)) {
								WRITE_PKT(pkt_ptr, it);
								cc->hit ++;
								goto this_job_end;
							} else {
								mprint(DEBUG, "Key does not match!\n");
							}
						#else
							if (pkt_ptr - pkt_base_ptr + it->nkey + it->nbytes + 6 > ETHERNET_MAX_FRAME_LEN) { 
								goto exceed_packet_len;
							}

							WRITE_PKT(pkt_ptr, it);
							cc->hit ++;
							goto this_job_end;
						#endif
						}
					}

					if (pkt_ptr - pkt_base_ptr + 1 > ETHERNET_MAX_FRAME_LEN) {
						goto exceed_packet_len;
					}
					/* We do not find one */
					*(uint8_t *)pkt_ptr = PROTOCOL_GET_NOT_FOUND;
					pkt_ptr += sizeof(uint8_t);
					cc->miss ++;

this_job_end:
					this_job->index ++;
					this_job->num --;
				} /* end of construct packet */

exceed_packet_len:
				total_length = pkt_ptr - pkt_base_ptr;
				/* One packet has finished construction, update chunk info */
				if (total_length > config->eiu_hdr_len) {
				#if !defined(LOCAL_TEST)
					(tx_mbufs.m_table[pkt_mark])->pkt.pkt_len
						= (tx_mbufs.m_table[pkt_mark])->pkt.data_len
						= total_length;
				#endif
					pkt_mark ++;

					/* total_length (payload) + IP_HEADER_LENGTH + UDP_HEADER_LENGTH */
					iph->total_length = rte_cpu_to_be_16((uint16_t)(total_length - sizeof(struct ether_hdr))); 
					udph->dgram_len = rte_cpu_to_be_16((uint16_t)(total_length - sizeof(struct ether_hdr) - sizeof(struct ipv4_hdr)));
					cc->total_bytes += total_length;
				} else {
					mprint(ERROR, "[sender] error in constructing packet, length %d\n", total_length);
				}
			} /* end of construct chunk */

#if defined(LOCAL_TEST)
			cc->total_packets += config->io_batch_num;
#else
			/* send the chunk */
			tx_mbufs.len = pkt_mark;
			ret = rte_eth_tx_burst(ifindex, (uint16_t)queue_id, tx_mbufs.m_table, (uint16_t)tx_mbufs.len);
			cc->total_packets += ret;
			cc->dropped_packets += tx_mbufs.len - ret;
#endif
		}
#endif
/* =================================================================================================*/

		assert(batch->delay != -1);
		/* Process insert buf */
#if 0
		if (batch->delay == 0) {
			for (i = 0; i < buf->num_insert_job; i ++) {
				mega_job_insert_t *job = &(buf->insert_job_list[i]); 
				/* Here we do not care about the results of insert. Just set the flags from
				 * ITEM_WRITING to 0. Because we believe that the chance for conflicting
				 * signature is really small, and the evicting process will handle it.
				 * The primary reason that we make this simple is to avoid concurrent 
				 * operations with the receiver, avoiding locks and sophisticated logic
				 * on the critical code path */
				if (job->it->flags != ITEM_WRITING) {
					mprint(ERROR, "[sender] flag is not ITEM_WRITING for insert job, i = %d\n", i);
				}
				job->it->flags = 0;
			}
		}
#endif

		/* Give available buffer */
		mega_sender_give_available_buffer(id);
	}

	return 0;
}

void *mega_sender_main(mega_sender_context_t *context)
{
	mega_sender_init(context);
	mprint(INFO, "[Sender %d] on core %d is sending via if:queue %d:%d ...\n", 
			context->unique_id, context->core_id, context->ifindex, context->queue_id);
	mega_sender_sendloop(context->ifindex, context->queue_id, context->unique_id);

	exit(0);
}
