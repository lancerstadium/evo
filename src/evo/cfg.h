

#ifndef _EVO_CFG_H_
#define _EVO_CFG_H_

#include <sob/sob.h>
#include <util/util.h>
#include <stdint.h>

// ==================================================================================== //
//                                    evo: typedef
// ==================================================================================== //

typedef uint8_t     u8;
typedef uint16_t    u16;
typedef uint32_t    u32;
typedef uint64_t    u64;
typedef int16_t     i16;
typedef int32_t     i32;
typedef int64_t     i64;
typedef int8_t      i8;
typedef float       f32;
typedef double      f64;
typedef void*       ptr;



// ==================================================================================== //
//                                    evo: config
// ==================================================================================== //

#define CFG_MODE_EMU
// #define CFG_MODE_ITP
// #define CFG_MODE_AOT
// #define CFG_MODE_JIT
// #define CFG_MODE_HYB

#define CFG_SISA        RV
#define CFG_SISA_BIT    32
#define CFG_SISA_RV
#define CFG_SISA_RVI
#define CFG_SISA_RVZifencei
#define CFG_SISA_RVZicsr
// #define CFG_SISA_RVM
// #define CFG_SISA_RVA
// #define CFG_SISA_RVF
// #define CFG_SISA_RVD
// #define CFG_SISA_RVQ

#define CFG_IISA        EIR
#define CFG_IISA_BIT    64
#define CFG_IISA_EIR

#define CFG_TISA        ARM
#define CFG_TISA_BIT    64
#define CFG_TISA_ARM
#if !defined(CFG_MODE_EMU) && !defined(CFG_TISA)
#ifdef __i386__
#define CFG_TISA        X86
#define CFG_TISA_BIT    32
#define CFG_TISA_X86

#elif  __x86_64__
#define CFG_TISA        X86
#define CFG_TISA_BIT    64
#define CFG_TISA_X86
#elif  __arm__
#define CFG_TISA        ARM
#define CFG_TISA_BIT    32
#define CFG_TISA_ARM
#elif  __aarch64__
#define CFG_TISA        ARM
#define CFG_TISA_BIT    64
#define CFG_TISA_ARM
#elif  __riscv__
#define CFG_TISA        RV
#endif // __ARCH__
#endif // CFG_TISA


#define CFG_WORD_SIZE       __WORDSIZE
#define CFG_CODE_CAP        1024
#define CFG_MEM_BASE        0x80000000
#define CFG_MEM_CAP         1024 * 1024
#define CFG_STACK_CAP       1024
#define CFG_GEN_ELF         "out.elf"      

#define CFG_TRANS_CAP       200
// 1e3: us, 1e6: ms, 1e9: s
#define CFG_PERF_TIMES      1e6 

#if CFG_WORD_SIZE == 64
typedef u64  PAddr;
#define PFMT "0x%016lx"
#define UFMT "%lu"
#elif CFG_WORD_SIZE == 32
typedef u32  PAddr;
#define PFMT "0x%08x"
#define UFMT "%u"
#endif

#define BFMT "0x%02x"
#define HFMT "0x%04x"
#define WFMT "0x%08x"
#define DFMT "0x%16lx"

#endif // _EVO_CFG_H_