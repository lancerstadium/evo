

#ifndef _EVO_REG_H_
#define _EVO_REG_H_

#include <evo/typ.h>


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

#endif // _EVO_REG_H_