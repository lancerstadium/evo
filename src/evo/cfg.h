

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

typedef struct {
    char n[32];     // name
    size_t w;       // bit width
    bool e;         // false: little endian, true: big endian
} Arch;

#define Arch_width(A)       ((A).w)
#define Arch_name(A)        ((A).n)
#define Arch_endian(A)      ((A).e)

typedef enum {
    UNKNOWN_ARCH,
    I386_ARCH,
    X86_64_ARCH,
    ARM_ARCH,
    AARCH64_ARCH,
    RISCV32_ARCH,
    ARCHID_SIZE
} ArchID;


UNUSED static Arch arch_map[ARCHID_SIZE] = {
    [UNKNOWN_ARCH]  = {"unknown",  0, false},
    [I386_ARCH]     = {"i386"   , 32, false},
    [X86_64_ARCH]   = {"x86_64" , 64, false},
    [RISCV32_ARCH]  = {"riscv32", 32, false}
};

#define Arch_def(N, U)

// ==================================================================================== //
//                                    evo: config
// ==================================================================================== //

#define CFG_SRC_ISA         RISCV32_ARCH

#ifndef CFG_TRG_ISA
#ifdef __i386__
#define CFG_TRG_ISA         I386_ARCH
#elif  __x86_64__
#define CFG_TRG_ISA         X86_64_ARCH
#elif  __arm__
#define CFG_TRG_ARM         ARM_ARCH
#elif  __aarch64__
#define CFG_TRG_AARCH64     AARCH64_ARCH
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