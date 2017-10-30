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

#ifndef MEGA_CONTEXT_H
#define MEGA_CONTEXT_H

#include "mega_batch.h"

typedef struct mega_sender_context_s {
	mega_batch_t *batch;
	int core_id;
	int queue_id;
	int initialized;
	int unique_id;
	int ifindex;
} mega_sender_context_t;

typedef struct mega_receiver_context_s {
	mega_batch_t *batch;
	int core_id;
	int queue_id;
	int initialized;
	int unique_id;
	int ifindex;
} mega_receiver_context_t;

typedef struct mega_scheduler_context_s {
	mega_batch_t *cpu_batch_set;
	int core_id; /* which core should gpu worker run */
	/* Add more info passing to GPU worker here ... */
} mega_scheduler_context_t;

void *mega_scheduler_main(mega_scheduler_context_t *context);
void *mega_receiver_main(mega_receiver_context_t *context);
void *mega_sender_main(mega_sender_context_t *context);

#endif
