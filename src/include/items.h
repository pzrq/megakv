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

#ifndef _MEGA_ITEM_H_
#define _MEGA_ITEM_H_


/*
#define ITEM_UPDATE_INTERVAL 60
*/

#define ITEM_FREE		1
#define ITEM_WRITING	2
#define ITEM_USING		3

#define ITEM_key(it) ((char *)it + sizeof(item))
#define ITEM_value(it) ((char *)it + sizeof(item) + it->nkey)

typedef struct _stritem {
#if defined(RWLOCK)
	pthread_rwlock_t rwlock;
#endif

	void *			next;	/* thread-local linked list */
	uint8_t         flags;   /* ITEM_* above */
	uint8_t         slabs_clsid;/* which slab class we're in */
	uint16_t        nkey;       /* key length, w/terminating null and padding */
	uint32_t        nbytes;     /* size of data */
	
	uint32_t		loc; /* The location of the item in slab, slab_num + offset */

	/* following this header, there will be keys and values */
} item;

item *item_alloc_batch(const uint32_t kvsize);

#endif
