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
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/queue.h>
#include <setjmp.h>
#include <stdarg.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <sched.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>

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
#include <rte_ethdev.h>
#include <rte_ring.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_byteorder.h>

#include "zipf.h"
#include "benchmark.h"

#define _GNU_SOURCE
#define __USE_GNU

#define MBUF_SIZE (2048 + sizeof(struct rte_mbuf) + RTE_PKTMBUF_HEADROOM)
#define NB_MBUF  2048

/*
 * RX and TX Prefetch, Host, and Write-back threshold values should be
 * carefully set for optimal performance. Consult the network
 * controller's datasheet and supporting DPDK documentation for guidance
 * on how these parameters should be set.
 */
#define RX_PTHRESH 8 /**< Default values of RX prefetch threshold reg. */
#define RX_HTHRESH 8 /**< Default values of RX host threshold reg. */
#define RX_WTHRESH 4 /**< Default values of RX write-back threshold reg. */

/*
 * These default values are optimized for use with the Intel(R) 82599 10 GbE
 * Controller and the DPDK ixgbe PMD. Consider using other values for other
 * network controllers and/or network drivers.
 */
#define TX_PTHRESH 36 /**< Default values of TX prefetch threshold reg. */
#define TX_HTHRESH 0  /**< Default values of TX host threshold reg. */
#define TX_WTHRESH 0  /**< Default values of TX write-back threshold reg. */

#define MAX_PKT_BURST 1
#define BURST_TX_DRAIN_US 100 /* TX drain every ~100us */

/*
 * Configurable number of RX/TX ring descriptors
 */
#define RTE_TEST_RX_DESC_DEFAULT 128
#define RTE_TEST_TX_DESC_DEFAULT 512
static uint16_t nb_rxd = RTE_TEST_RX_DESC_DEFAULT;
static uint16_t nb_txd = RTE_TEST_TX_DESC_DEFAULT;

struct mbuf_table {
	unsigned len;
	struct rte_mbuf *m_table[MAX_PKT_BURST];
};

#define MAX_RX_QUEUE_PER_LCORE 16
#define MAX_TX_QUEUE_PER_PORT 16
struct lcore_queue_conf {
	struct mbuf_table tx_mbufs[MAX_TX_QUEUE_PER_PORT];
} __rte_cache_aligned;
struct lcore_queue_conf lcore_queue_conf[NUM_QUEUE];

static const struct rte_eth_conf port_conf = {
	.rxmode = {
		.mq_mode = ETH_MQ_RX_RSS,
		.max_rx_pkt_len = ETHER_MAX_LEN,
		.split_hdr_size = 0,
		.header_split   = 0, /**< Header Split disabled */
		.hw_ip_checksum = 0, /**< IP checksum offload disabled */
		.hw_vlan_filter = 0, /**< VLAN filtering disabled */
		.jumbo_frame    = 0, /**< Jumbo Frame Support disabled */
		.hw_strip_crc   = 0, /**< CRC stripped by hardware */
	},
	.rx_adv_conf = {
		.rss_conf = {
			.rss_key = NULL,
			.rss_hf = ETH_RSS_IP,
		},
	},
	.txmode = {
		.mq_mode = ETH_MQ_TX_NONE,
	},
};

static const struct rte_eth_rxconf rx_conf = {
	.rx_thresh = {
		.pthresh = RX_PTHRESH,
		.hthresh = RX_HTHRESH,
		.wthresh = RX_WTHRESH,
	},
};

static const struct rte_eth_txconf tx_conf = {
	.tx_thresh = {
		.pthresh = TX_PTHRESH,
		.hthresh = TX_HTHRESH,
		.wthresh = TX_WTHRESH,
	},
	.tx_free_thresh = 0, /* Use PMD default values */
	.tx_rs_thresh = 0, /* Use PMD default values */
	/*
	 * As the example won't handle mult-segments and offload cases,
	 * set the flag by default.
	 */
	.txq_flags = ETH_TXQ_FLAGS_NOMULTSEGS | ETH_TXQ_FLAGS_NOOFFLOADS,
};

struct rte_mempool *recv_pktmbuf_pool[NUM_QUEUE];
struct rte_mempool *send_pktmbuf_pool = NULL;
#define NUM_MAX_CORE 32
/* Per-port statistics struct */
struct benchmark_core_statistics {
	uint64_t tx;
	uint64_t rx;
	uint64_t dropped;
	int enable;
} __rte_cache_aligned;
struct benchmark_core_statistics core_statistics[NUM_MAX_CORE];

/* A tsc-based timer responsible for triggering statistics printout */
#define TIMER_MILLISECOND 2000000ULL /* around 1ms at 2 Ghz */
#define MAX_TIMER_PERIOD 86400 /* 1 day max */
static int64_t timer_period = 5 * TIMER_MILLISECOND * 1000; /* default period is 5 seconds */

struct timeval startime;
struct timeval endtime;
uint64_t ts_count[NUM_QUEUE], ts_total[NUM_QUEUE];

typedef struct context_s {
	unsigned int core_id;
	unsigned int queue_id;
} context_t;


void *rx_loop(context_t *);
void *tx_loop(context_t *);

#define EIU_HEADER_LEN	42//14+20+8 = 42
#define ETHERNET_HEADER_LEN	14

/* 1500 bytes MTU + 14 Bytes Ethernet header */
int pktlen;

#if defined(PRELOAD)
int loading_mode = 1;
#endif

/* Print out statistics on packets dropped */
static void
print_stats(void)
{
	uint64_t total_packets_dropped, total_packets_tx, total_packets_rx;
	uint64_t total_latency = 0, total_latency_cnt = 0;
	unsigned core_id, queue_id;

	total_packets_dropped = 0;
	total_packets_tx = 0;
	total_packets_rx = 0;

	const char clr[] = { 27, '[', '2', 'J', '\0' };
	const char topLeft[] = { 27, '[', '1', ';', '1', 'H','\0' };

	/* Clear screen and move to top left */
	printf("%s%s", clr, topLeft);

	struct timeval subtime;
	gettimeofday(&endtime, NULL);
	timersub(&endtime, &startime, &subtime);

	printf("\nPort statistics ====================================");
	printf("\nNUM_QUEUE=%d, WORKLOAD_ID=%d, LOAD_FACTOR=%.2f", NUM_QUEUE, WORKLOAD_ID, LOAD_FACTOR);
	printf(", DIS_ZIPF %.2f", ZIPF_THETA);

	for (core_id = 0; core_id < NUM_MAX_CORE; core_id ++) {
		if (core_statistics[core_id].enable == 0) continue;
		printf("\nStatistics for core %d ------------------------------"
				"    Packets sent: %11"PRIu64
				"    Packets received: %11"PRIu64
				"    Packets dropped: %11"PRIu64,
				core_id,
				core_statistics[core_id].tx,
				core_statistics[core_id].rx,
				core_statistics[core_id].dropped);

		total_packets_dropped += core_statistics[core_id].dropped;
		total_packets_tx += core_statistics[core_id].tx;
		total_packets_rx += core_statistics[core_id].rx;

		core_statistics[core_id].dropped = 0;
		core_statistics[core_id].tx = 0;
		core_statistics[core_id].rx = 0;
	}

	for (queue_id = 0; queue_id < NUM_QUEUE; queue_id ++) {
		total_latency += ts_total[queue_id];
		total_latency_cnt += ts_count[queue_id];
		ts_total[queue_id] = 0;
		ts_count[queue_id] = 1;
	}
	printf("\nAggregate statistics ==============================="
			"\nTotal packets sent: %18"PRIu64
			"\nTotal get sent: %22"PRIu64
			"\nTotal set sent: %22"PRIu64
			"\nTotal packets received: %14"PRIu64
			"\nTotal packets dropped: %15"PRIu64,
			total_packets_tx,
			total_packets_tx * number_packet_get[WORKLOAD_ID],
			total_packets_tx * number_packet_set[WORKLOAD_ID],
			total_packets_rx,
			total_packets_dropped);
	printf("\nTX Speed = %5.2lf Gbps, RX Speed = %5.2lf Gbps, latency count %18"PRIu64 " average %lf",
			(double)(total_packets_tx * pktlen * 8) / (double) ((subtime.tv_sec*1000000+subtime.tv_usec) * 1000),
			(double)(total_packets_rx * pktlen * 8) / (double) ((subtime.tv_sec*1000000+subtime.tv_usec) * 1000),
			total_latency_cnt, (total_latency/total_latency_cnt)/(rte_get_tsc_hz()/1e6));
	printf("\nGET request speed = %5.2lf MOPS, SET request speed = %5.2lf MOPS\n",
			total_packets_tx * number_packet_get[WORKLOAD_ID] / (double) (subtime.tv_sec*1000000+subtime.tv_usec),
			total_packets_tx * number_packet_set[WORKLOAD_ID] / (double) (subtime.tv_sec*1000000+subtime.tv_usec));
	printf("\n====================================================\n");

	gettimeofday(&startime, NULL);
}

/* main processing loop */
void *tx_loop(context_t *context)
{
	struct rte_mbuf *m;
	unsigned i, k;
	struct lcore_queue_conf *qconf;
	unsigned int core_id = context->core_id;
	unsigned int queue_id = context->queue_id;

	unsigned long mask = 1 << core_id;
	if (sched_setaffinity(0, sizeof(unsigned long), (cpu_set_t *)&mask) < 0) {
		printf("core id = %d\n", core_id);
		assert(0);
	}

	qconf = &lcore_queue_conf[queue_id];

	unsigned int tmp_pktlen;

	struct ether_hdr *ethh;
	struct ipv4_hdr *iph;
	struct udp_hdr *udph;

	/* for 1GB hash table, 512MB signature, 32bits, total is 128M = 2^29/2^2 = 2^27
	 * load 80% of the hash table */
	const uint32_t total_cnt = (uint32_t)TOTAL_CNT;
	uint32_t preload_cnt = (uint32_t)PRELOAD_CNT;

	struct zipf_gen_state zipf_state;
	mehcached_zipf_init(&zipf_state, (uint64_t)preload_cnt - 2, (double)ZIPF_THETA, (uint64_t)21);
	//printf("LOAD_FACTOR is %f, total key cnt is %d\n", LOAD_FACTOR, total_cnt);

	char *ptr;

	for (i = 0; i < MAX_PKT_BURST; i ++) {
		m = rte_pktmbuf_alloc(send_pktmbuf_pool);
		assert (m != NULL);
		m->pkt.nb_segs = 1;
		m->pkt.next = NULL;
		qconf->tx_mbufs[queue_id].m_table[i] = m;

		ethh = (struct ether_hdr *)rte_pktmbuf_mtod(m, unsigned char *);
		//ethh->s_addr = LOCAL_MAC_ADDR;
		ethh->ether_type = rte_cpu_to_be_16((uint16_t)(ETHER_TYPE_IPv4));

		iph = (struct ipv4_hdr *)((unsigned char *)ethh + sizeof(struct ether_hdr));
		iph->version_ihl = 0x40 | 0x05;
		iph->type_of_service = 0;
		iph->packet_id = 0;
		iph->fragment_offset = 0;
		iph->time_to_live = 64;
		iph->next_proto_id = IPPROTO_UDP;
		iph->hdr_checksum = 0;
		iph->src_addr = LOCAL_IP_ADDR;
		iph->dst_addr = KV_IP_ADDR;

		udph = (struct udp_hdr *)((unsigned char *)iph + sizeof(struct ipv4_hdr));
		udph->src_port = LOCAL_UDP_PORT;
		udph->dst_port = KV_UDP_PORT;
		udph->dgram_cksum = 0;

		ptr = (char *)rte_ctrlmbuf_data(m) + EIU_HEADER_LEN;
		*(uint16_t *)ptr = PROTOCOL_MAGIC;
	}

	qconf->tx_mbufs[queue_id].len = MAX_PKT_BURST;


	struct rte_mbuf **m_table;
	uint32_t *ip;
	uint32_t ip_ctr = 1;
	unsigned int port, ret;
	uint32_t get_key = 1, set_key = 1;


#if defined(PRELOAD)

	/* update packet length for 100% SET operations in PRELOAD */
	pktlen = 1510;

	for (i = 0; i < MAX_PKT_BURST; i ++) {
		m = qconf->tx_mbufs[queue_id].m_table[i];
		assert (m != NULL);
		rte_pktmbuf_pkt_len(m) = (uint16_t)pktlen;
		rte_pktmbuf_data_len(m) = (uint16_t)pktlen;

		ethh = (struct ether_hdr *)rte_pktmbuf_mtod(m, unsigned char *);
		iph = (struct ipv4_hdr *)((unsigned char *)ethh + sizeof(struct ether_hdr));
		udph = (struct udp_hdr *)((unsigned char *)iph + sizeof(struct ipv4_hdr));

		iph->total_length = rte_cpu_to_be_16((uint16_t)(pktlen - sizeof(struct ether_hdr)));
		udph->dgram_len = rte_cpu_to_be_16((uint16_t)(pktlen - sizeof(struct ether_hdr) - sizeof(struct ipv4_hdr)));
	}

	uint32_t payload_len;
	if (queue_id == 0) {
		printf("Going to insert %u keys, LOAD_FACTOR is %.2f\n", preload_cnt, LOAD_FACTOR);

		/* preload the keys */
		//while (set_key < NUM_DEFINED_GET * 0.01 * total_cnt) {
		while (set_key < preload_cnt) {
			m_table = (struct rte_mbuf **)qconf->tx_mbufs[queue_id].m_table;

			/* construct a send buffer */
			for (i = 0; i < qconf->tx_mbufs[queue_id].len; i ++) {
				ip = (uint32_t *)((char *)rte_ctrlmbuf_data(m_table[i]) + 26);
				*ip = ip_ctr ++;

				/* skip the packet header and magic number */
				ptr = (char *)rte_ctrlmbuf_data(m_table[i]) + EIU_HEADER_LEN + MEGA_MAGIC_NUM_LEN;
				/* basic length = header len + magic len + ending mark len */
				payload_len = EIU_HEADER_LEN + MEGA_MAGIC_NUM_LEN + MEGA_END_MARK_LEN;

				/* construct a packet */
				/* ----------------------------------------------------- */
				while (payload_len + SET_LEN <= ETHERNET_MAX_FRAME_LEN) {
					*(uint16_t *)ptr = MEGA_JOB_SET;
					ptr += sizeof(uint16_t); /* 2 bytes job type */
					*(uint16_t *)ptr = KEY_LEN;
					ptr += sizeof(uint16_t); /* 2 bytes key length */
					*(uint32_t *)ptr = VALUE_LEN;
					ptr += sizeof(uint32_t); /* 4 bytes value length */

					/* 64 bits key */
					if (BITS_INSERT_BUF == 0)
						*(uint32_t *)(ptr + sizeof(uint32_t)) = set_key;
					else
						*(uint32_t *)(ptr + sizeof(uint32_t)) = (rte_bswap32(set_key & 0xff) << (8 - BITS_INSERT_BUF)) | (set_key);
					*(uint32_t *)(ptr) = set_key;

					ptr += KEY_LEN;
					ptr += VALUE_LEN;

					payload_len += SET_LEN;

					set_key ++;
					if (set_key >= preload_cnt) {
						break;
					}
				}

				assert(payload_len < ETHERNET_MAX_FRAME_LEN);
				/* write the ending mark */
				*(uint16_t *)ptr = 0xFFFF;

				/* reduce insert speed */
				int k = 20000;
				while(k > 0) k--;
			}

			port = 0;
			assert(qconf->tx_mbufs[queue_id].len == MAX_PKT_BURST);
			ret = rte_eth_tx_burst(port, (uint16_t)queue_id, m_table, (uint16_t)qconf->tx_mbufs[queue_id].len);
		}

		printf(" ==========================     Hash table has been loaded     ========================== \n");

		loading_mode = 0;
	}

	while (loading_mode == 1) ;

	/* Different receivers use different keys start point */
	get_key = (10000 * queue_id) % preload_cnt;
	set_key = preload_cnt + queue_id * ((total_cnt - preload_cnt)/NUM_QUEUE);
#endif

	/* update packet length for the workload packets */
	pktlen = length_packet[WORKLOAD_ID];

	for (i = 0; i < MAX_PKT_BURST; i ++) {
		m = qconf->tx_mbufs[queue_id].m_table[i];
		assert (m != NULL);
		rte_pktmbuf_pkt_len(m) = (uint16_t)pktlen;
		rte_pktmbuf_data_len(m) = (uint16_t)pktlen;

		ethh = (struct ether_hdr *)rte_pktmbuf_mtod(m, unsigned char *);
		iph = (struct ipv4_hdr *)((unsigned char *)ethh + sizeof(struct ether_hdr));
		udph = (struct udp_hdr *)((unsigned char *)iph + sizeof(struct ipv4_hdr));

		iph->total_length = rte_cpu_to_be_16((uint16_t)(pktlen - sizeof(struct ether_hdr)));
		udph->dgram_len = rte_cpu_to_be_16((uint16_t)(pktlen - sizeof(struct ether_hdr) - sizeof(struct ipv4_hdr)));
	}

	if (queue_id == 0) {
		gettimeofday(&startime, NULL);
	}
	core_statistics[core_id].enable = 1;
	/* NOW SEARCH AND INSERT */
	// ===========================================================
	while (1) {
		assert (qconf->tx_mbufs[queue_id].len == MAX_PKT_BURST);
		m_table = (struct rte_mbuf **)qconf->tx_mbufs[queue_id].m_table;
		for (i = 0; i < qconf->tx_mbufs[queue_id].len; i ++) {
			ip = (uint32_t *)((char *)rte_ctrlmbuf_data(m_table[i]) + 26);
			*ip = ip_ctr ++;

			/* skip the packet header and magic number */
			ptr = (char *)rte_ctrlmbuf_data(m_table[i]) + EIU_HEADER_LEN + MEGA_MAGIC_NUM_LEN;

			for (k = 0; k < number_packet_get[WORKLOAD_ID]; k ++) {
				*(uint16_t *)ptr = MEGA_JOB_GET;
				/* skip job_type, key length = 4 bytes in total */
				ptr += sizeof(uint16_t);
				*(uint16_t *)ptr = KEY_LEN;
				ptr += sizeof(uint16_t);

				get_key = (uint32_t)mehcached_zipf_next(&zipf_state) + 1;
				assert(get_key >= 1 && get_key <= preload_cnt);

				/* here we try to evenly distribute the key through insert bufs,
				 * on the first 32 bits, the highest 5 bits are used for 32 insert bufs,
				 * htonl(key & 0xff) << 3 is to assign the 5 bits.
				 * We also need to distribute keys among buckets, and it is the lower
				 * bits are used for hash. the "|key" is setting the hash.
				 * The next 32 bits are used as signature, just key ++ */
				if (BITS_INSERT_BUF == 0)
					*(uint32_t *)(ptr + sizeof(uint32_t)) = get_key;
				else
					*(uint32_t *)(ptr + sizeof(uint32_t)) = (rte_bswap32(get_key & 0xff) << (8 - BITS_INSERT_BUF)) | get_key;
				*(uint32_t *)(ptr) = get_key;

				ptr += KEY_LEN;
			}

			for (k = 0; k < number_packet_set[WORKLOAD_ID]; k ++) {
				*(uint16_t *)ptr = MEGA_JOB_SET;
				ptr += sizeof(uint16_t);
				*(uint16_t *)ptr = KEY_LEN;
				ptr += sizeof(uint16_t);
				*(uint32_t *)ptr = VALUE_LEN;
				ptr += sizeof(uint32_t);

				set_key ++;
#if defined(PRELOAD)
				if (set_key >= preload_cnt + (queue_id + 1) * ((total_cnt - preload_cnt)/NUM_QUEUE)) {
					// FIXME
					assert(0);
				}
#else
				if (set_key >= total_cnt) {
					set_key = 1;
				}
#endif

				if (BITS_INSERT_BUF == 0)
					*(uint32_t *)(ptr + sizeof(uint32_t)) = set_key;
				else
					*(uint32_t *)(ptr + sizeof(uint32_t)) = (rte_bswap32(set_key & 0xff) << (8 - BITS_INSERT_BUF)) | (set_key);
				*(uint32_t *)(ptr) = set_key;

				ptr += KEY_LEN;
				ptr += VALUE_LEN;
			}
			//total_cnt += number_packet_set[WORKLOAD_ID];

			*(uint16_t *)ptr = 0xFFFF;
		}

		for (i = 0; i < qconf->tx_mbufs[queue_id].len; i ++) {
			/* use an IP field for measuring latency, disabled  */
			//*(uint64_t *)((char *)rte_ctrlmbuf_data(m_table[i]) + ETHERNET_HEADER_LEN + 4) = rte_rdtsc_precise();
			if (rte_pktmbuf_pkt_len(m) != length_packet[WORKLOAD_ID]) {
				printf("%d != %d\n", rte_pktmbuf_pkt_len(m), length_packet[WORKLOAD_ID]);
				assert(0);
			}
		}

		port = 0;
		assert(qconf->tx_mbufs[queue_id].len == MAX_PKT_BURST);
		ret = rte_eth_tx_burst(port, (uint16_t) queue_id, m_table, (uint16_t) qconf->tx_mbufs[queue_id].len);
		core_statistics[core_id].tx += ret;
		if (unlikely(ret < qconf->tx_mbufs[queue_id].len)) {
			core_statistics[core_id].dropped += (qconf->tx_mbufs[queue_id].len - ret);
		}
	}
}

/* main processing loop */
void *rx_loop(context_t *context)
{
	struct rte_mbuf *pkts_burst[MAX_PKT_BURST];
	struct rte_mbuf *m;
	unsigned int core_id = context->core_id;
	unsigned int queue_id = context->queue_id;
	uint64_t prev_tsc, diff_tsc, cur_tsc, timer_tsc;
	unsigned portid, nb_rx;

	unsigned long mask = 1 << core_id;
	if (sched_setaffinity(0, sizeof(unsigned long), (cpu_set_t *)&mask) < 0) {
		assert(0);
	}

	prev_tsc = 0;
	timer_tsc = 0;

	core_statistics[core_id].enable = 1;

	while (1) {

		cur_tsc = rte_rdtsc();
		diff_tsc = cur_tsc - prev_tsc;

		/* if timer is enabled */
		if (timer_period > 0) {
			/* advance the timer */
			timer_tsc += diff_tsc;
			/* if timer has reached its timeout */
			if (unlikely(timer_tsc >= (uint64_t) timer_period)) {
				/* do this only on master core */
			#if defined(PRELOAD)
				if (queue_id == 0 && loading_mode == 0) {
			#else
				if (queue_id == 0) {
			#endif
					print_stats();
					/* reset the timer */
					timer_tsc = 0;
				}
			}
		}
		prev_tsc = cur_tsc;

		/*
		 * Read packet from RX queues
		 */

		portid = 0;
		nb_rx = rte_eth_rx_burst((uint8_t) portid, queue_id, pkts_burst, MAX_PKT_BURST);

		core_statistics[core_id].rx += nb_rx;

		if (nb_rx > 0) {
			m = pkts_burst[0];
			rte_prefetch0(rte_pktmbuf_mtod(m, void *));

			//uint64_t now = rte_rdtsc_precise();
			uint64_t now = rte_rdtsc();
			uint64_t ts = *(uint64_t *)((char *)rte_ctrlmbuf_data(m) + ETHERNET_HEADER_LEN + 4);
			if (ts != 0) {
				ts_total[queue_id] += now - ts;
				ts_count[queue_id] ++;
			}
		}

		if (nb_rx > 0) {
			unsigned k = 0;
			do {
				rte_pktmbuf_free(pkts_burst[k]);
			} while (++k < nb_rx);
		}
	}
}

/* Check the link status of all ports in up to 9s, and print them finally */
static void
check_all_ports_link_status(uint8_t port_num, uint32_t port_mask)
{
#define CHECK_INTERVAL 100 /* 100ms */
#define MAX_CHECK_TIME 90 /* 9s (90 * 100ms) in total */
	uint8_t portid, count, all_ports_up, print_flag = 0;
	struct rte_eth_link link;

	printf("\nChecking link status");
	fflush(stdout);
	for (count = 0; count <= MAX_CHECK_TIME; count++) {
		all_ports_up = 1;
		for (portid = 0; portid < port_num; portid++) {
			if ((port_mask & (1 << portid)) == 0)
				continue;
			memset(&link, 0, sizeof(link));
			rte_eth_link_get_nowait(portid, &link);
			/* print link status if flag set */
			if (print_flag == 1) {
				if (link.link_status)
					printf("Port %d Link Up - speed %u "
						"Mbps - %s\n", (uint8_t)portid,
						(unsigned)link.link_speed,
				(link.link_duplex == ETH_LINK_FULL_DUPLEX) ?
					("full-duplex") : ("half-duplex\n"));
				else
					printf("Port %d Link Down\n",
						(uint8_t)portid);
				continue;
			}
			/* clear all_ports_up flag if any link down */
			if (link.link_status == 0) {
				all_ports_up = 0;
				break;
			}
		}
		/* after finally printing all link status, get out */
		if (print_flag == 1)
			break;

		if (all_ports_up == 0) {
			printf(".");
			fflush(stdout);
			rte_delay_ms(CHECK_INTERVAL);
		}

		/* set the print_flag if all ports up or timeout */
		if (all_ports_up == 1 || count == (MAX_CHECK_TIME - 1)) {
			print_flag = 1;
			printf("done\n");
		}
	}
}

int
MAIN(int argc, char **argv)
{
	int ret;
	int i;
	uint8_t nb_ports;
	uint8_t portid, queue_id;

	/* init EAL */
	int t_argc = 5;
	char *t_argv[] = {"./build/benchmark", "-c", "f", "-n", "1"};
	ret = rte_eal_init(t_argc, t_argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "Invalid EAL arguments\n");

	char str[10];
	/* create the mbuf pool */
	for (i = 0; i < NUM_QUEUE; i ++) {
		sprintf(str, "%d", i);
		recv_pktmbuf_pool[i] =
			rte_mempool_create(str, NB_MBUF,
					MBUF_SIZE, 32,
					sizeof(struct rte_pktmbuf_pool_private),
					rte_pktmbuf_pool_init, NULL,
					rte_pktmbuf_init, NULL,
					rte_socket_id(), 0);
		if (recv_pktmbuf_pool[i] == NULL)
			rte_exit(EXIT_FAILURE, "Cannot init mbuf pool\n");
	}

	send_pktmbuf_pool =
		rte_mempool_create("send_mbuf_pool", NB_MBUF,
				   MBUF_SIZE, 32,
				   sizeof(struct rte_pktmbuf_pool_private),
				   rte_pktmbuf_pool_init, NULL,
				   rte_pktmbuf_init, NULL,
				   rte_socket_id(), 0);
	if (send_pktmbuf_pool == NULL)
		rte_exit(EXIT_FAILURE, "Cannot init mbuf pool\n");

	if (rte_eal_pci_probe() < 0)
		rte_exit(EXIT_FAILURE, "Cannot probe PCI\n");

	nb_ports = rte_eth_dev_count();
	assert (nb_ports == 1);

	/* Initialise each port */
	for (portid = 0; portid < nb_ports; portid++) {
		/* init port */
		printf("Initializing port %u... ", (unsigned) portid);
		ret = rte_eth_dev_configure(portid, NUM_QUEUE, NUM_QUEUE, &port_conf);
		if (ret < 0)
			rte_exit(EXIT_FAILURE, "Cannot configure device: err=%d, port=%u\n",
				  ret, (unsigned) portid);

		for (queue_id = 0; queue_id < NUM_QUEUE; queue_id ++) {
			/* init RX queues */
			ret = rte_eth_rx_queue_setup(portid, queue_id, nb_rxd,
					rte_eth_dev_socket_id(portid), &rx_conf,
					recv_pktmbuf_pool[queue_id]);
			if (ret < 0)
				rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup:err=%d, port=%u\n",
						ret, (unsigned) portid);

			/* init TX queues */
			ret = rte_eth_tx_queue_setup(portid, queue_id, nb_txd,
					rte_eth_dev_socket_id(portid), &tx_conf);
			if (ret < 0)
				rte_exit(EXIT_FAILURE, "rte_eth_tx_queue_setup:err=%d, port=%u\n",
						ret, (unsigned) portid);
		}

		/* Start device */
		ret = rte_eth_dev_start(portid);
		if (ret < 0)
			rte_exit(EXIT_FAILURE, "rte_eth_dev_start:err=%d, port=%u\n",
				  ret, (unsigned) portid);

		printf("done: \n");

		rte_eth_promiscuous_enable(portid);

		/* initialize port stats */
		memset(&core_statistics, 0, sizeof(core_statistics));
	}
	fflush(stdout);

	check_all_ports_link_status(nb_ports, 0);

	for (i = 0; i < NUM_QUEUE; i ++) {
		ts_total[i] = 0;
		ts_count[i] = 1;
	}

	pthread_t tid;
	pthread_attr_t attr;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

	context_t *context;

	for (i = 0; i < NUM_QUEUE; i ++) {
		if (i == 0) {
			context = (context_t *) malloc (sizeof(context_t));
			context->core_id = 0;
			context->queue_id = i;
			if (pthread_create(&tid, &attr, (void *)rx_loop, (void *)context) != 0) {
				perror("pthread_create error!!\n");
			}
		}

		context = (context_t *) malloc (sizeof(context_t));
#if defined(AFFINITY_MAX_QUEUE)
		context->core_id = i + 1;
#elif defined(AFFINITY_ONE_NODE)
		context->core_id = i * 2 + 1;
#else
		printf("No affinity\n");
		exit(0);
#endif
		context->queue_id = i;
		if (pthread_create(&tid, &attr, (void *)tx_loop, (void *)context) != 0) {
			perror("pthread_create error!!\n");
		}
	}

	while (1) {
		sleep(10);
	}

	return 0;
}

