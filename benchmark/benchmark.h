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

#ifndef _MAIN_H_
#define _MAIN_H_

#ifdef RTE_EXEC_ENV_BAREMETAL
#define MAIN _main
#else
#define MAIN main
#endif

/* Following protocol speicific parameters should be same with MEGA */
#define PROTOCOL_MAGIC	0x1234
#define MEGA_JOB_GET 0x2
#define MEGA_JOB_SET 0x3
/* BITS_INSERT_BUF should be same with mega: config->bits_insert_buf */
#define BITS_INSERT_BUF 3 // 2^3 = 8

#define MEGA_MAGIC_NUM_LEN	2
#define MEGA_END_MARK_LEN	2

/* ========== Key definitions ========== */

/* if PRELOAD is disabled, the main program should preload locally */
//#define PRELOAD		1

/* Key Distribution, only one can be enabled */
#define DIS_UNIFORM	1
//#define DIS_ZIPF	1

#if defined(DIS_ZIPF)
	#define ZIPF_THETA 0.99
	#define AFFINITY_MAX_QUEUE 1
/* Core Affinity, zipf distribution needs more cores for calculation */
	#define NUM_QUEUE 7
#elif defined(DIS_UNIFORM)
	#define ZIPF_THETA 0.00
	#define AFFINITY_ONE_NODE 1
	#define NUM_QUEUE 4
#endif

/* Hash Table Load Factor, These should be the same with the main program 
 * if PRELOAD is disabled! TODO: avoid mismatches */
#define LOAD_FACTOR 0.2
#define PRELOAD_CNT (LOAD_FACTOR * ((1 << 30)/8))
#define TOTAL_CNT (((uint32_t)1 << 31) - 1)


#define KEY_LEN			8
#define VALUE_LEN		8
#define SET_LEN		(KEY_LEN + VALUE_LEN + 8)
#define ETHERNET_MAX_FRAME_LEN	1514

/* choose which workload to use with the above parameters 
 * 0 - 100% GET, 1 - 95% GET, 5% SET
 * Only supports 8 byte key/value */
int WORKLOAD_ID = 0;

/* 100%GET : 0  Set, 100Get, (42+2+2) + (12 * 122) = 1510
 * 95% GET : 5  Set, 95 Get, (42+2+2) + (24*1 + 12*19)*5 = 1306
 * 90% GET : 11 Set, 99 Get, (42+2+2) + (24*1 + 12*9)*11 = 1498
 * 80% GET : 20 Set, 80 Get, (42+2+2) + (24*1 + 12*4)*20 = 1486
 * 70% GET : 27 Set, 63 Get, (42+2+2) + (24*3 + 12*7)*9  = 1450
 * 60% GET : 34 Set, 51 Get, (42+2+2) + (24*2 + 12*3)*17 = 1474
 * 50% GET : 40 Set, 40 Get, (42+2+2) + (24*1 + 12*1)*40 = 1486
 * */
const unsigned int number_packet_set[8] = {0, 5, 11, 20, 27, 34, 40};
const unsigned int number_packet_get[8] = {122, 95, 99, 80, 63, 51, 40};
const unsigned int length_packet[8] = {1510, 1306, 1498, 1486, 1450, 1474, 1486};

/* ------------------------------------------------------- */

/* TODO: Set following values.
 * DPDK does not require MAC address to send a packet */
//#define LOCAL_MAC_ADDR 
#define KV_IP_ADDR (uint32_t)(789)
#define KV_UDP_PORT (uint16_t)(124)
#define LOCAL_IP_ADDR (uint32_t)(456)
#define LOCAL_UDP_PORT (uint16_t)(123)

#endif /* _MAIN_H_ */
