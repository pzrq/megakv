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

#ifndef MEGA_CONFIG_H
#define MEGA_CONFIG_H

#include <rte_ether.h>

typedef struct mega_config_s {
	unsigned int cpu_worker_num;
	unsigned int scheduler_num;
	unsigned int num_queue_per_port;

	unsigned int eiu_hdr_len;
	unsigned int ethernet_hdr_len;

	// Most important argument for realtime scheduling algorithm
	unsigned int I; // 40ms, 30ms ...

	unsigned int io_batch_num;
	int ifindex0;
	int ifindex1;

	unsigned int num_insert_buf;
	unsigned int bits_insert_buf;

	unsigned int batch_max_search_job;
	unsigned int batch_max_insert_job;
	unsigned int batch_max_delete_job;
	unsigned int perslab_bits;
	unsigned int slabclass_max_elem_num;
	unsigned int loc_bits;
	unsigned int slab_id_bits;
	
	unsigned int GPU_search_thread_num;
	unsigned int GPU_delete_thread_num;
	unsigned int GPU_threads_per_blk;
	unsigned int item_max_size;

	size_t mem_limit;
	double factor;
	int prealloc;
	int evict;
	unsigned int evict_batch_size;

	uint32_t local_ip_addr;
	uint16_t local_udp_port;
	struct ether_addr local_mac_addr;
} mega_config_t;

#endif
