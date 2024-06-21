

#ifndef _EVO_INS_H_
#define _EVO_INS_H_

#include <evo/typ.h>


#ifdef __cplusplus
extern "C" {
#endif


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
        const char* code; \
        TypeVec tv;       \
    } InsnDef(T)

#define InsnDef_def(T, ...)                                   \
    InsnDef_T(T);                                             \
    static InsnDef(T) InsnTbl(T)[InsnMax(T)] = {__VA_ARGS__}; \
    void InsnDef_OP_def(T, displayone)(char* res, size_t i) { \
        if (i < InsnMax(T)) {                                 \
            sprintf(res, "%s", InsnTbl(T)[i].name);           \
        }                                                     \
    }                                                         \
    void InsnDef_OP_def(T, display)(char* res) {              \
        for (size_t i = 0; i < InsnMax(T); i++) {             \
            sprintf(res, "%s\n", InsnTbl(T)[i].name);         \
        }                                                     \
    }

#define InsnDef_display(T, res)  InsnDef_OP(T, display)(res)
#define InsnDef_displayone(T, res, i)  InsnDef_OP(T, displayone)(res, i)


#ifdef __cplusplus
}
#endif  // __cplusplus

#endif // _EVO_INS_H_