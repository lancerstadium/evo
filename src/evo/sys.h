/**
 * =================================================================================== //
 * @file sys.h
 * @author lancerstadium (lancerstadium@163.com)
 * @brief system header file
 * @version 0.1
 * @date 2024-07-03
 * @copyright Copyright (c) 2024
 * =================================================================================== //
 */

// ==================================================================================== //
//                                       evo/sys.h
// ==================================================================================== //

#ifndef __EVO_SYS_H__
#define __EVO_SYS_H__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ==================================================================================== //
//                                        include
// ==================================================================================== //

#include <stdint.h>
#include <stdarg.h>
#include <stdlib.h>

#if __APPLE__
#include <sys/malloc.h>
#include <sys/errno.h>
#else
#include <malloc.h>
#endif

#include "lock.h"

// ==================================================================================== //
//                                      host sys port
// ==================================================================================== //

void* sys_malloc(size_t size);
void sys_free(void* ptr);
void* sys_realloc(void* ptr, size_t size);



// ==================================================================================== //
//                                      host cpu info
// ==================================================================================== //

int cpu_check();
int cpu_get_mask(int mask);
int cpu_set_affine(int mask);
int cpu_get_cluster_mask(int cluster);



// ==================================================================================== //
//                                       string operate
// ==================================================================================== //

char* sys_strdup(const char* src);


#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_SYS_H__