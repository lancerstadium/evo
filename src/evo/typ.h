
#ifndef _EVO_TYP_H_
#define _EVO_TYP_H_

#include <evo/cfg.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif


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

#if EVO_WORD_SIZE == 64
typedef u64 vaddr;
#else
typedef u32 vaddr;
#endif

// ==================================================================================== //
//                                    evo: Width
// ==================================================================================== //

typedef union {
    u8 as_u8;
    i8 as_i8;
} Byte;

typedef union {
    u16 as_u16;
    i16 as_i16;
} Half;

typedef union {
    u32 as_u32;
    i32 as_i32;
    f32 as_f32;
#if EVO_WORD_SIZE == 32
    ptr as_ptr;
#endif
} Word;

typedef union {
    u64 as_u64;
    i64 as_i64;
    f64 as_f64;
#if EVO_WORD_SIZE == 64
    ptr as_ptr;
#endif
} Dword;


Word Word_u32_new(u32 val);
Word Word_i32_new(i32 val);
Word Word_f32_new(f32 val);
#if EVO_WORD_SIZE == 32
Word Word_ptr_new(ptr val);
#endif

Dword Dword_u64_new(u64 val);
Dword Dword_i64_new(i64 val);
Dword Dword_f64_new(f64 val);
#if EVO_WORD_SIZE == 64
Dword Dword_ptr_new(ptr val);
#endif

#define Wsize(W) sizeof(W)

// ==================================================================================== //
//                                    evo: Type
// ==================================================================================== //




typedef enum {
    TYPE_ANY = 0,
    TYPE_FLT,
    TYPE_INT_S,
    TYPE_INT_U,
    TYPE_MEM_ADDR,
    TYPE_INS_ADDR,
    TYPE_STK_ADDR,
    TYPE_SYM_ID,
    TYPE_BOOL,
    TYPE_SIZE
} Type;


#ifdef __cplusplus
}
#endif  // __cplusplus

#endif // _EVO_TYP_H_