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
#include <assert.h>

#include "libgpuhash.h"
#include "mega_common.h"

#ifdef USE_SSE
#include <nmmintrin.h>
#endif

inline uint64_t calc_signature(char *key, int key_length)
{
	// TODO: 1) utilize all key bits, 2) try aes

	//assert(key_length >= 8);

#if 0
	uint64_t r = 0;
	uint64_t *k = (uint64_t *)key;

	/* If the key length can not be divided by 8,
	 * we discard the left 1~7 bytes for the crc64 */
	while (key_length >= 8) {
#ifdef USE_SSE
		r = _mm_crc32_u64(r, *k);
#else
		r = r ^ (*k);
#endif
		k ++;
		key_length -= 8;
	}
#else
	uint64_t r = *(uint64_t *)key;
#endif

	/* we cannot have a 0 signature, use 1 to break the tie */
	if(r == 0) r = 1;

	return r;
}

const char *MEGA_PRINT_MSG[PRINT_LEVELS] = {
		" stat",
		"fatal",
		"error",
		" warn",
		" info",
		"debug"
};

static void show_stackframe(void) {
  void *trace[32];
  char **messages = (char **)NULL;
  int i, trace_size = 0;

  trace_size = backtrace(trace, 32);
  messages = backtrace_symbols(trace, trace_size);
  fprintf(stderr, "Printing stack frames:\n");
  for (i=0; i < trace_size; ++i)
        fprintf(stderr, "\t%s\n", messages[i]);
}

void panic(char *msg)
{
	fprintf(stderr, "[mega panic] %s\n", msg);
	show_stackframe();
	exit(-1);
}

#ifdef MEGA_PRINT_BUFFER
char *mprint_buffer = NULL;
#define FBUFFER_SIZE		(128L * 1000L * 1000L)
int mprint_lines = 0;
int mprint_head = 0;
struct spinlock mprint_lock;
#endif

void mprint_init(void)
{
#ifdef MEGA_PRINT_BUFFER
	int i;

	initlock(&mprint_lock);

	mprint_buffer = (char *)malloc(FBUFFER_SIZE);
	if (!mprint_buffer) {
		fprintf(stderr, "failed to initialize mprint buffer\n");
		exit(-1);
	}

	for (i = 0; i < FBUFFER_SIZE; i += 4096)
		mprint_buffer[i] = 'x';
#endif
}

void mprint_fini(void)
{
#ifdef MEGA_PRINT_BUFFER
	int i, head = 0, len;
	if (mprint_buffer) {
		for (i = 0; i < mprint_lines; i++) {
			len = printf("%s", mprint_buffer + head);
			head += len + 1;
		}
		free(mprint_buffer);
		mprint_buffer = NULL;
	}
#endif
}
