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

#ifndef MEGA_LOG_H
#define MEGA_LOG_H

typedef struct mega_log_sample_s {
	unsigned int	isMsg;
	unsigned int	isErr;
	double        timer;
	unsigned int  nbytes;
	int           loops;
	char *        fmt;
	char *        msg;
	double           num;
} mega_log_sample_t;

typedef struct mega_log_s {
	unsigned int idx;
	unsigned int loops;
	unsigned int loop_entries;
	unsigned int loop_timers;
 
	mega_log_sample_t *samples;
} mega_log_t;

void mega_log_init(mega_log_t *log);
void mega_log_loop_marker(mega_log_t *log);
void mega_log_msg(mega_log_t *log, const char *format, const char *msg, const double num);
void mega_log_timer(mega_log_t *log, const char *format, const char *msg, double timer, unsigned int nbytes, int loops);
void mega_log_print(mega_log_t *log);
#endif
