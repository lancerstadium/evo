
#ifndef _EVO_EVO_H_
#define _EVO_EVO_H_

#include <evo/cfg.h>
#include <gen/gen.h>

#ifdef __cplusplus
extern "C" {
#endif


// ==================================================================================== //
//                                    evo: Addr & Width
// ==================================================================================== //

typedef struct {
    union {
        u8 as_u8;
        i8 as_i8;
    };
} Byte;
typedef struct {
    union {
        u16 as_u16;
        i16 as_i16;
    };
} Half;
typedef struct {
    union {
        u32 as_u32;
        i32 as_i32;
        f32 as_f32;
    };
} Word;
typedef struct {
    union {
        u64 as_u64;
        i64 as_i64;
        f64 as_f64;
        ptr as_ptr;
    };
} Dword;

#define Width_OP(W, T, OP)          CONCAT3(W##_, T##_, OP)
#define Width_OP_def(W, T, OP)      UNUSED Width_OP(W, T, OP)

#define Width_def(W, T)                \
    W Width_OP_def(W, T, new)(T val) { \
        return (W){                    \
            .as_##T = val};            \
    }

Width_def(Word, u32);
Width_def(Word, i32);
Width_def(Word, f32);
Width_def(Dword, u64);
Width_def(Dword, i64);
Width_def(Dword, f64);
Width_def(Dword, ptr);

// ==================================================================================== //
//                                    evo: Byte Map
// ==================================================================================== //

typedef struct {
    u8* val;
    size_t size;
} ByteVec;

#define ByVec(...)  { .val = (u8[]){__VA_ARGS__}, .size = (sizeof((u8[]){__VA_ARGS__}) / sizeof(Byte)) }
#define ByN0(N)    { .val = {0}, .size = (N) }

#define ByU8(V) \
    {   .val = (u8[]){   \
        (u8)((V) >>  0), \
        }, .size = 1 }

#define ByU16(V) \
    {   .val = (u8[]){   \
        (u8)((V) >>  0), \
        (u8)((V) >>  8), \
        }, .size = 2 }

#define ByU32(V) \
    {   .val = (u8[]){   \
        (u8)((V) >>  0), \
        (u8)((V) >>  8), \
        (u8)((V) >> 16), \
        (u8)((V) >> 24), \
        }, .size = 4 }

#define ByU64(V) \
    {   .val = (u8[]){  \
        (u8)((V) >>  0), \
        (u8)((V) >>  8), \
        (u8)((V) >> 16), \
        (u8)((V) >> 24), \
        (u8)((V) >> 32), \
        (u8)((V) >> 38), \
        (u8)((V) >> 46), \
        (u8)((V) >> 54), \
        }, size = 8}

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
    TYPE_BOOL       = 1 << 8,
} Type;


typedef struct {
    Type* tys;
    size_t size;
} TypeVec;


#define TyVec(...) { .tys = (Type[]){__VA_ARGS__}, .size = (sizeof((Type[]){__VA_ARGS__}) / sizeof(Type)) }


// ==================================================================================== //
//                                    evo: Val
// ==================================================================================== //

typedef struct {
    TypeVec t;
    ByteVec b;
} Val;


// ==================================================================================== //
//                                    evo: Reg
// ==================================================================================== //

#define RegID(T)      CONCAT(RegID_, T)
#define RegID_T(T, ...) \
    typedef enum {      \
        __VA_ARGS__     \
        T##_REGID_SIZE  \
    } RegID(T)

#define RegID_def(T, ...) \
    RegID_T(T, __VA_ARGS__)

#define RegMax(T)              T##_REGID_SIZE
#define RegTbl(T)              T##_reg_tbl
#define RegDef(T)              CONCAT(RegDef_, T)
#define RegDef_OP(T, OP)       CONCAT3(RegDef_, T ## _, OP)
#define RegDef_OP_def(T, OP)   UNUSED RegDef_OP(T, OP)
#define RegDef_T(T)        \
    typedef struct {       \
        RegID(T) id;       \
        const char* name;  \
        const char* alias; \
    } RegDef(T)

#define RegDef_def(T, ...)                                                \
    RegDef_T(T);                                                          \
    static RegDef(T) RegTbl(T)[RegMax(T)] = {__VA_ARGS__};                \
    void RegDef_OP_def(T, displayone)(char* res, size_t i) {              \
        if (i < RegMax(T)) {                                              \
            sprintf(res, "%d: %s", RegTbl(T)[i].id, RegTbl(T)[i].name);   \
        }                                                                 \
    }                                                                     \
    void RegDef_OP_def(T, display)(char* res) {                           \
        for (size_t i = 0; i < RegMax(T); i++) {                          \
            sprintf(res, "%d: %s\n", RegTbl(T)[i].id, RegTbl(T)[i].name); \
        }                                                                 \
    }

#define RegDef_display(T, res)  RegDef_OP(T, display)(res)
#define RegDef_displayone(T, res, i)  RegDef_OP(T, displayone)(res, i)


// ==================================================================================== //
//                                    evo: Insn
// ==================================================================================== //

#define InsnID(T)               CONCAT(InsnID_, T)
#define InsnID_T(T, ...) \
    typedef enum {       \
        __VA_ARGS__      \
        T##_INSNID_SIZE  \
    } InsnID(T)

#define InsnID_def(T, ...) \
    InsnID_T(T, __VA_ARGS__)

#define InsnMax(T)              T##_INSNID_SIZE
#define InsnTbl(T)              T##_insn_tbl
#define InsnDef(T)              CONCAT(InsnDef_, T)
#define InsnDef_OP(T, OP)       CONCAT3(InsnDef_, T ## _, OP)
#define InsnDef_OP_def(T, OP)   UNUSED InsnDef_OP(T, OP)
#define InsnDef_T(T)      \
    typedef struct {      \
        InsnID(T) id;     \
        const char* name; \
        ByteVec bc;       \
        TypeVec tv;       \
    } InsnDef(T)

#define InsnDef_def(T, ...)                                                        \
    InsnDef_T(T);                                                                  \
    static InsnDef(T) InsnTbl(T)[InsnMax(T)] = {__VA_ARGS__};                      \
    void InsnDef_OP_def(T, displayone)(char* res, size_t i) {                      \
        if (i < InsnMax(T)) {                                                      \
            sprintf(res, "%-14s %s", ByHex(InsnTbl(T)[i].bc), InsnTbl(T)[i].name); \
        }                                                                          \
    }                                                                              \
    void InsnDef_OP_def(T, display)(char* res) {                                   \
        for (size_t i = 0; i < InsnMax(T); i++) {                                  \
            sprintf(res, "%s\n", InsnTbl(T)[i].name);                              \
        }                                                                          \
    }

#define InsnDef_display(T, res) InsnDef_OP(T, display)(res)
#define InsnDef_displayone(T, res, i)  InsnDef_OP(T, displayone)(res, i)

#define Insn(T) CONCAT(Insn_, T)
#define Insn_OP(T, OP) CONCAT3(Insn_, T##_, OP)
#define Insn_OP_def(T, OP) UNUSED Insn_OP(T, OP)
#define Insn_T(T, S)  \
    typedef struct {  \
        InsnID(T) id; \
        S             \
    } Insn(T)

// ==================================================================================== //
//                                    evo: Task
// ==================================================================================== //


#define TaskCtx(T) CONCAT(TaskCtx_, T)
#define TaskCtx_OP(T, OP) CONCAT3(TaskCtx_, T##_, OP)
#define TaskCtx_OP_def(T, OP) UNUSED TaskCtx_OP(T, OP)
#define TaskCtx_T(T, S) \
    typedef struct {    \
        S               \
    } TaskCtx(T)

#define TaskCtx_def(T, S) \
    TaskCtx_T(T, S);

#define Task(T) CONCAT(Task_, T)
#define Task_OP(T, OP) CONCAT3(Task_, T##_, OP)
#define Task_OP_def(T, OP) UNUSED Task_OP(T, OP)
#define Task_T(T)            \
    typedef struct Task(T) { \
        const char* name;    \
        TaskCtx(T) ctx;      \
    } Task(T)

#define Task_def(T, S, ...)                        \
    TaskCtx_def(T, S);                             \
    Task_T(T);                                     \
    __VA_ARGS__                                    \
    Task(T) * Task_OP_def(T, create)(char* name) { \
        Task(T)* t = malloc(sizeof(Task(T)));      \
        t->name = name;                            \
        TaskCtx_OP(T, init)(&t->ctx);              \
        return t;                                  \
    }                                              \
    void Task_OP_def(T, run)(Task(T) * t) {        \
        TaskCtx_OP(T, run)(&t->ctx);               \
    }

#define Task_str(T) STR(T)
#define Task_create(T, name) Task_OP(T, create)(name)
#define Task_run(T, t) Task_OP(T, run)(t)


Task_def(Dump,
    ElfCtx *elf;
,
    void TaskCtx_OP_def(Dump, init) (TaskCtx(Dump) *ctx) {
        ctx->elf = ElfCtx_init();
    }
    void TaskCtx_OP_def(Dump, elf) (TaskCtx(Dump) *ctx, char* name) {
        ElfCtx_gen(ctx->elf, name);
    }
    void TaskCtx_OP_def(Dump, run) (TaskCtx(Dump) *ctx) {
        TaskCtx_OP(Dump, elf)(ctx, "a.out");
    }
    void TaskCtx_OP_def(Dump, clean) (TaskCtx(Dump) *ctx) {
        ElfCtx_free(ctx->elf);
    }
);


// ==================================================================================== //
//                                    evo: CPU
// ==================================================================================== //

#define CPUState(T)             CONCAT(CPUState_, T)
#define CPUState_T(T, S)      \
    typedef struct {          \
        Task(Dump) task_dump; \
        S                     \
    } CPUState(T)

#define CPUState_def(T, S, ...) \
    CPUState_T(T, S); __VA_ARGS__



// ==================================================================================== //
//                                    evo: ISA
// ==================================================================================== //

#if defined(CFG_SISA_EIR) || defined(CFG_IISA_EIR) || defined(CFG_TISA_EIR)
#include <isa/eir/def.h>
#endif
#if defined(CFG_SISA_X86) || defined(CFG_IISA_X86) || defined(CFG_TISA_X86)
#include <isa/x86/def.h>
#endif
#if defined(CFG_SISA_ARM) || defined(CFG_IISA_ARM) || defined(CFG_TISA_ARM)
#include <isa/arm/def.h>
#endif
#if defined(CFG_SISA_RV)  || defined(CFG_IISA_RV)  || defined(CFG_TISA_RV)
#include <isa/rv/def.h>
#endif

#if defined(CFG_MODE_ITP) || defined(CFG_MODE_AOT) || defined(CFG_MODE_JIT) || defined(CFG_MODE_HYB)
typedef CONCAT(CPUState_, CFG_IISA) CPUState_0;
#elif defined(CFG_MODE_EMU)
typedef CONCAT(CPUState_, CFG_SISA) CPUState_0;
#else
#error Unsupport EVO_MODE, Config options: EMU / ITP / AOT / JIT / HYB 
#endif



#ifdef __cplusplus
}
#endif  // __cplusplus

#endif // _EVO_EVO_H_