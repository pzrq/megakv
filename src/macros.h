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


#define MEM_LIMIT			((size_t)1 << 34) // 2^34 = 16GB

#define EVICT_BATCH_SIZE	512

/* N queues = N receivers + N senders */
#define NUM_QUEUE_PER_PORT	7

#define MAX_WORKER_NUM		12

/* modify this according to system CPU, for accurate timer */
#define CPU_FREQUENCY_US	2600 // 2.6GHz/1e6

//#define TIME_MEASURE		1

/* number of scheduling cycles to call one insert and delete,
 * batch more insert and delete jobs to improve performance,
 * set to 0 will be the same with original form */
#define NUM_SCHED_INS_DEL	0


#if defined(PREFETCH_PIPELINE)
	#define PREFETCH_DISTANCE	2
#endif

#if defined(PREFETCH_BATCH)
	/* different key-value sizes may use different prefetch
	 * distance to maximize performance */
	#define PREFETCH_BATCH_DISTANCE	5
#endif

/* ========== Following should also be set with network ========== */

/* define KEY and VALUE length, for preload and setting max key/value length */
#define KVSIZE	0

#if (KVSIZE == 0)
	#define KEY_LEN				8
	#define VALUE_LEN			8
	#define LOAD_FACTOR			0.2
#elif (KVSIZE == 1)
	#define KEY_LEN				16
	#define VALUE_LEN			64
	#define LOAD_FACTOR			0.1
#elif (KVSIZE == 2)
	#define KEY_LEN				32
	#define VALUE_LEN			512
	#define LOAD_FACTOR			0.01
#elif (KVSIZE == 3)
	#define KEY_LEN				128
	#define VALUE_LEN			1024
	#define LOAD_FACTOR			0.01
#endif

#define GET_LEN		(KEY_LEN + 4)
#define SET_LEN		(KEY_LEN + VALUE_LEN + 8)


/* ========== Following definition only in LOCAL_TEST ========== */
#if defined(LOCAL_TEST)

#define GET100				1
//#define GET95				1
//#define GET50				1

#if defined(GET100)
	#define NUM_DEFINED_GET 100
	#define NUM_DEFINED_SET 0
#elif defined(GET95)
	#define NUM_DEFINED_GET 95
	#define NUM_DEFINED_SET 5
#elif defined(GET50)
	#define NUM_DEFINED_GET 50
	#define NUM_DEFINED_SET 50
#endif

#if defined(DIS_ZIPF)
	#define ZIPF_THETA	0.99
#elif defined(DIS_UNIFORM)
	#define ZIPF_THETA	0.00
#endif

#endif // LOCAL_TEST
