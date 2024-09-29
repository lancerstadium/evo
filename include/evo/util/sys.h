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
//                                     evo/util/sys.h
// ==================================================================================== //

#ifndef __EVO_UTIL_SYS_H__
#define __EVO_UTIL_SYS_H__

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

#include <evo/config.h>


// ==================================================================================== //
//                                      host sys port
// ==================================================================================== //

void* sys_malloc        (size_t size);
void  sys_free          (void* ptr);
void* sys_realloc       (void* ptr, size_t size);


// ==================================================================================== //
//                                       string operate
// ==================================================================================== //

char* sys_strdup        (const char* src);
char* sys_get_file_ext  (const char* path);
char* sys_get_file_name (const char* path);
char* sys_memory_size   (int size);

// ==================================================================================== //
//                                       system time
// ==================================================================================== //

double sys_time         ();

// ==================================================================================== //
//                                       system device
// ==================================================================================== //

int sys_has_avx         ();
int sys_has_avx_vnni    ();
int sys_has_avx2        ();
int sys_has_avx512      ();
int sys_has_avx512_vbmi ();
int sys_has_avx512_vnni ();
int sys_has_avx512_bf16 ();
int sys_has_fma         ();
int sys_has_neon        ();
int sys_has_sve         ();
int sys_has_arm_fma     ();
int sys_has_metal       ();
int sys_has_f16c        ();
int sys_has_fp16_va     ();
int sys_has_wasm_simd   ();
int sys_has_blas        ();
int sys_has_cuda        ();
int sys_has_vulkan      ();
int sys_has_kompute     ();
int sys_has_gpublas     ();
int sys_has_sse3        ();
int sys_has_ssse3       ();
int sys_has_riscv_v     ();
int sys_has_sycl        ();
int sys_has_rpc         ();
int sys_has_vsx         ();
int sys_has_matmul_int8 ();
int sys_has_cann        ();
int sys_has_llamafile   ();

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_UTIL_SYS_H__