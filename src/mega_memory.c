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

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#include "mega_memory.h"
#include "mega_macros.h"

ALLOCSZ_ATTR(1)
void *mega_mem_malloc(const size_t size)
{
	void *aux = malloc(size);

	if (mega_unlikely(!aux && size)) {
		perror("malloc");
		return NULL;
	}

	return aux;
}

ALLOCSZ_ATTR(1)
void *mega_mem_calloc(const size_t size)
{
	void *buf = calloc(1, size);
	if (mega_unlikely(!buf)) {
		return NULL;
	}

	return buf;
}

ALLOCSZ_ATTR(2)
void *mega_mem_realloc(void *ptr, const size_t size)
{
	void *aux = realloc(ptr, size);

	if (mega_unlikely(!aux && size)) {
		perror("realloc");
		return NULL;
	}

	return aux;
}

void mega_mem_free(void *ptr)
{
	free(ptr);
}
