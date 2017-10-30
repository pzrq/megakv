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

#ifndef MEGA_COMMON_H
#define MEGA_COMMON_H

#include <stdio.h>
#include <execinfo.h>

#define STAT	0
#define FATAL	1
#define ERROR	2
#define WARN	3
#define INFO	4
#define DEBUG	5
#define PRINT_LEVELS	6

#ifndef MEGA_PRINT_LEVEL
#define MEGA_PRINT_LEVEL		INFO
#endif

extern const char *MEGA_PRINT_MSG[];

#ifdef MEGA_PRINT_BUFFER
#include <sys/time.h>
#include "spinlock.h"

extern char *mprint_buffer;
extern int mprint_lines;
extern int mprint_head;
extern struct spinlock mprint_lock;

#define mprint(lvl, fmt, arg...) \
	do { \
		if (lvl <= MEGA_PRINT_LEVEL) { \
			int len; \
			struct timeval t; \
			gettimeofday(&t, NULL); \
			acquire(&mprint_lock); \
			len = sprintf(mprint_buffer + mprint_head, \
					"[%d %s] %lf " fmt, gettid(), MEGA_PRINT_MSG[lvl], \
					((double)(t.tv_sec) + t.tv_usec / 1000000.0), ##arg); \
			mprint_lines++; \
			mprint_head += len + 1; \
			release(&mprint_lock); \
		} \
	} while (0)
#else
#define mprint(lvl, fmt, arg...) \
		do { \
			if (lvl <= MEGA_PRINT_LEVEL) { \
				printf("[%d %s] " fmt, getpid(), MEGA_PRINT_MSG[lvl], ##arg); \
			} \
		} while (0)
#endif

void mprint_init(void);
void mprint_fini(void);

#ifndef gettid
#include <unistd.h>
#include <sys/syscall.h>
static inline pid_t gettid(void)
{
	return (pid_t)syscall(186);
}
#endif

void panic(char *msg);
uint64_t calc_signature(char *key, int key_length);

#define PROTOCOL_MAGIC			0x1234
#define PROTOCOL_VALUE			0x0001
#define PROTOCOL_GET_NOT_FOUND	0x0002

#define ETHERNET_MAX_FRAME_LEN	1514

#define MAX_GPU_STREAM 16

#endif
