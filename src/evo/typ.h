
#ifndef _EVO_TYP_H_
#define _EVO_TYP_H_

#include <evo/cfg.h>


#ifdef __cplusplus
extern "C" {
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
} Word;
typedef union {
    u64 as_u64;
    i64 as_i64;
    f64 as_f64;
    ptr as_ptr;
} Dword;


Word Word_u32_new(u32 val);
Word Word_i32_new(i32 val);
Word Word_f32_new(f32 val);
Dword Dword_u64_new(u64 val);
Dword Dword_i64_new(i64 val);
Dword Dword_f64_new(f64 val);
Dword Dword_ptr_new(ptr val);


#define Wsize(W) sizeof(W)

typedef struct {
    u8* val;
    size_t size;
} ByteVec;

#define ByVec(...) { .val = (u8[]){__VA_ARGS__}, .size = (sizeof((u8[]){__VA_ARGS__}) / sizeof(Byte)) }
#define ByU32(V) \
    {   .val = (u8[]){   \
        (u8)((V) >>  0), \
        (u8)((V) >>  8), \
        (u8)((V) >> 16), \
        (u8)((V) >> 24), \
        }, .size = 4 }


char* ByHex(ByteVec v) {
    char* tmp = malloc((3 + v.size * 4)* sizeof(char));
    snprintf(tmp, 3, "0x");
    for(size_t i = 0; i < v.size; i++) {
        char hex[4];
        if(v.val[i]) {
            snprintf(hex, 4, "%02x ", v.val[i]);
        } else {
            snprintf(hex, 4, "00 ");
        }
        strcat(tmp, hex);
    }
    return tmp;
}



// ==================================================================================== //
//                                    evo: Type
// ==================================================================================== //


typedef enum {
    TYPE_ANY        = 0,
    TYPE_FLOAT      = 1 << 0,
    TYPE_SINT       = 1 << 1,
    TYPE_UINT       = 1 << 2,
    TYPE_MEM_ADDR   = 1 << 3,
    TYPE_INSN_ADDR  = 1 << 4,
    TYPE_STARK_ADDR = 1 << 5,
    TYPE_REG_ID     = 1 << 6,
    TYPE_SYMBOL_ID  = 1 << 7,
    TYPE_BOOL       = 1 << 8
} Type;


typedef struct {
    Type* tys;
    size_t size;
} TypeVec;


#define TyVec(...) { .tys = (Type[]){__VA_ARGS__}, .size = (sizeof((Type[]){__VA_ARGS__}) / sizeof(Type)) }



#ifdef __cplusplus
}
#endif  // __cplusplus

#endif // _EVO_TYP_H_