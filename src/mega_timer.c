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

#include "mega_timer.h"
#include "macros.h"

#include <sys/time.h>
#include <time.h>
#include <emmintrin.h>


static inline uint64_t read_tsc(void)
{
	union {
		uint64_t tsc_64;
		struct {
			uint32_t lo_32;
			uint32_t hi_32;
		};
	} tsc;
	//_mm_mfence();
	asm volatile("rdtsc" :
				"=a" (tsc.lo_32),
				"=d" (tsc.hi_32));
	return tsc.tsc_64;
}

int mega_timer_init(mega_timer_t *timer)
{
	timer->start = 0;
	timer->clocks = 0;

	return 0;
}

int mega_timer_start(mega_timer_t *timer)
{
	timer->start = read_tsc();

	return 0;
}

int mega_timer_restart(mega_timer_t *timer)
{
	timer->start = read_tsc();
	timer->clocks = 0;

	return 0;
}

int mega_timer_stop(mega_timer_t *timer)
{
	//timer->start = 0;
	timer->clocks += (read_tsc() - timer->start);

	return 0;
}

int mega_timer_reset(mega_timer_t *timer)
{
	return mega_timer_init(timer);
}

double mega_timer_get_total_time(mega_timer_t *timer)
{
	//returns microsecond as unit -- second * 1,000,000
	return (double)(timer->clocks) / (double) CPU_FREQUENCY_US;
}

double mega_timer_get_elapsed_time(mega_timer_t *timer)
{
	return (double)(read_tsc() - timer->start) / (double) CPU_FREQUENCY_US;
}

