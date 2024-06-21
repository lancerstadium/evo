

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


#define CFG_SISA(C)      CONCAT(CFG_SISA_, C)
#define CFG_TISA(C)      CONCAT(CFG_TISA_, C)


#ifndef CFG_TRG_ISA_NAME
#ifdef __i386__

#elif  __x86_64__

#elif  __arm__

#elif  __aarch64__

#endif  // __ARCH__
#endif  // CFG_ARG_ISA



#define CFG_WORD_SIZE       __WORDSIZE
#define CFG_CODE_CAP        1024
#define CFG_MEM_BASE        0x8000000
#define CFG_MEM_CAP         1024
#define CFG_STACK_CAP       1024
#define CFG_GEN_ELF         "out.elf"
#define CFG_ISA(A)          CONCAT(CFG_ISA_, A)
#define CFG_ISA_SIZE        

#if CFG_WORD_SIZE == 64
typedef u64  PAddr;
#define PFMT "0x%016lx"
#define UFMT "%lu"
#elif CFG_WORD_SIZE == 32
typedef u32  PAddr;
#define PFMT "0x%08x"
#define UFMT "%u"
#endif




#endif // _EVO_CFG_H_