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

#ifndef MEGA_JOB_H
#define MEGA_JOB_H

#include <stdint.h>
#include <rte_ether.h>
#include "items.h"
#include "../macros.h"

#define MEGA_JOB_GET 0x2
#define MEGA_JOB_SET 0x3

#define MAX_KEY_LENGTH   KEY_LEN
#define MAX_VALUE_LENGTH VALUE_LEN

#define MEGA_MAGIC_NUM_LEN 2 // sizeof(uint16_t)
#define MEGA_JOB_TYPE_LEN  2 // sizeof(uint16_t)
#define MEGA_KEY_SIZE_LEN  2 // sizeof(uint16_t)
#define MEGA_VALUE_SIZE_LEN  4 // sizeof(uint32_t)

typedef struct mega_job_search_s {
	struct ether_addr eth_addr;
	uint32_t ip;
	uint16_t port;

	uint16_t num;
	uint32_t index; /* the No. of this job in the batch */

#if defined(KEY_MATCH)
	char	 key[MAX_KEY_LENGTH];
	uint16_t nkey;
#endif
} mega_job_search_t;

typedef struct mega_job_insert_s {
	item *it; /* For set jobs, unset its flag */
} mega_job_insert_t;

#endif
