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

#ifndef MEGA_TIMER_H
#define MEGA_TIMER_H

#include <stdint.h>

/**
 * \file Timer.h
 * \brief A timer class that provides a cross platform timer for use
 * in timing code progress with a high degree of accuracy.
 * FIXME:
 * 1s = 1000ms (millisecond)
 * 1ms = 1000us (microsecond)
 * 1us = 1000ns (nanosecond)
 * this counter returns in terms of us
 */


typedef struct mega_timer_s {
    uint64_t clocks;
    uint64_t start;
} mega_timer_t;

int mega_timer_init(mega_timer_t *timer);
int mega_timer_start(mega_timer_t *timer);
int mega_timer_restart(mega_timer_t *timer);
int mega_timer_stop(mega_timer_t *timer);
int mega_timer_reset(mega_timer_t *timer);
double mega_timer_get_total_time(mega_timer_t *timer);
double mega_timer_get_elapsed_time(mega_timer_t *timer);

#endif

