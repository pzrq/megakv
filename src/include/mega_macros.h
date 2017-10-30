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

#ifndef MEGA_MACROS_H
#define MEGA_MACROS_H

#include <stdlib.h>

/* Boolean */
#define MEGA_FALSE 0
#define MEGA_TRUE  !MEGA_FALSE
#define MEGA_ERROR -1

/* Architecture */
#define INTSIZE sizeof(int)

/* Print macros */
#define MEGA_INFO     0x1000
#define MEGA_ERR      0X1001
#define MEGA_WARN     0x1002
#define MEGA_BUG      0x1003


//#define mega_info(...)  mega_print(MEGA_INFO, __VA_ARGS__)
//#define mega_err(...)   mega_print(MEGA_ERR, __VA_ARGS__)
//#define mega_warn(...)  mega_print(MEGA_WARN, __VA_ARGS__)
//#define mega_trace(...)  mega_print(MEGA_WARN, __VA_ARGS__)
#define mega_info  printf
#define mega_err   printf
#define mega_warn  printf
#define mega_trace  printf

/* Transport type */
#ifndef ARRAY_SIZE
# define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

#ifdef __GNUC__ /* GCC supports this since 2.3. */
 #define PRINTF_WARNINGS(a,b) __attribute__ ((format (printf, a, b)))
#else
 #define PRINTF_WARNINGS(a,b)
#endif

#ifdef __GNUC__ /* GCC supports this since 2.7. */
 #define UNUSED_PARAM __attribute__ ((unused))
#else
 #define UNUSED_PARAM
#endif

/*
 * Validation macros
 * -----------------
 * Based on article http://lwn.net/Articles/13183/
 *
 * ---
 * ChangeSet 1.803, 2002/10/18 16:28:57-07:00, torvalds@home.transmeta.com
 *
 *	Make a polite version of BUG_ON() - WARN_ON() which doesn't
 *	kill the machine.
 *
 *	Damn I hate people who kill the machine for no good reason.
 * ---
 *
 */

#define mega_unlikely(x) __builtin_expect((x),0)
#define mega_likely(x) __builtin_expect((x),1)
#define mega_prefetch(x, ...) __builtin_prefetch(x, __VA_ARGS__)

#define mega_is_bool(x) ((x == MEGA_TRUE || x == MEGA_FALSE) ? 1 : 0)

#define mega_bug(condition) do {                                          \
        if (mega_unlikely((condition)!=0)) {                              \
            mega_print(MEGA_BUG, "Bug found in %s() at %s:%d",              \
                     __FUNCTION__, __FILE__, __LINE__);                 \
            abort();                                                    \
        }                                                               \
    } while(0)

/*
 * Macros to calculate sub-net data using ip address and sub-net prefix
 */

#define MEGA_NET_IP_OCTECT(addr,pos) (addr >> (8 * pos) & 255)
#define MEGA_NET_NETMASK(addr,net) htonl((0xffffffff << (32 - net)))
#define MEGA_NET_BROADCAST(addr,net) (addr | ~MEGA_NET_NETMASK(addr,net))
#define MEGA_NET_NETWORK(addr,net) (addr & MEGA_NET_NETMASK(addr,net))
#define MEGA_NET_WILDCARD(addr,net) (MEGA_NET_BROADCAST(addr,net) ^ MEGA_NET_NETWORK(addr,net))
#define MEGA_NET_HOSTMIN(addr,net) net == 31 ? MEGA_NET_NETWORK(addr,net) : (MEGA_NET_NETWORK(addr,net) + 0x01000000)
#define MEGA_NET_HOSTMAX(addr,net) net == 31 ? MEGA_NET_BROADCAST(addr,net) : (MEGA_NET_BROADCAST(addr,net) - 0x01000000);

#if __GNUC__ >= 4
 #define MEGA_EXPORT __attribute__ ((visibility ("default")))
#else
 #define MEGA_EXPORT
#endif

// TRACE
#define MEGA_TRACE(...) do {} while (0)

#endif
