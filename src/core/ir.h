

#ifndef EVO_CORE_IR_H
#define EVO_CORE_IR_H

#include "../util/util.h"


// ==================================================================================== //
//                                   Defines
// ==================================================================================== //

#define IR_OPCODE(i)  IR_OPCODE_##i
#define IR_TYPE(i)    IR_TYPE_##i
#define IR_OPERAND(i) IR_OPERAND_##i
#define IR_ITEM(i)    IR_##i##_item
#define IR_BLK_NUM 5

#if UINTPTR_MAX == 0xffffffff
#define IR_PTR32 1
#define IR_PTR64 0
#elif UINTPTR_MAX == 0xffffffffffffffffu
#define IR_PTR32 0
#define IR_PTR64 1
#else
#error evo IR can work only for 32 or 64-bit targets
#endif
#define IR_MAX_SCALE UINT8_MAX
#define IR_MAX_REG_NUM UINT32_MAX
#define IR_NON_HARD_REG IR_MAX_REG_NUM


typedef uint8_t IRScale_t;                                  /* Index reg scale in memory */
typedef uint32_t IRReg_t;                                   /* Register (> 0): i, f, or (l)d  */
typedef int64_t IRDisp_t;                                   /* Address displacement in memory */
typedef struct IRItem *IRItem_t;                            /* IR 模块类型 */
typedef struct IRInsn *IRLabel_t;                           /* IR 指令标签：本质上是 IR 指令类型的地址 */
typedef struct IRInsn *IRInsn_t;                            /* IR 指令类型 */
typedef struct IRModule *IRModule_t;                        /* IR 模块类型 */
typedef const char *IRName_t;                               /* IR 名称类型 */
struct IRContext; 
typedef struct IRContext *IRContext_t;                      /* IR 上下文类型 */


// ==================================================================================== //
//                                   IR: Opcode
// ==================================================================================== //

/**
 * @brief evo IR 中间代码指令操作码
 * 
 * @note
 * evo IR：强类型的 IR，大多数指令都有一个目标操作数和一个或两个源操作数，目标操作数只能是寄存器或内存（不能是立即数）。
 * 
 * 约束：
 * 1. 每个寄存器只能包含一种类型的值
 * 2. 操作数类型应该符合指令的要求
 * 
 * 类型缩写:
 * - I  : int (64-bit)
 * - S  : short (32-bit)
 * - U  : unsigned
 * - F  : float (32-bit)
 * - D  : double (64-bit)
 * - LD : long double
 */
typedef enum {
    /** 
     * 两操作数数据控制指令（30）
     * 形式：[OP] <operand1>, <operand2>
     */ 
    REP4 (IR_OPCODE, MOV, FMOV, DMOV, LDMOV),                          // 操作数移动指令
    REP6 (IR_OPCODE, EXT8, EXT16, EXT32, UEXT8, UEXT16, UEXT32),       // 数据位拓展指令
    REP3 (IR_OPCODE, I2F, I2D, I2LD),                                  // 带符号整数转化为浮点数
    REP3 (IR_OPCODE, UI2F, UI2D, UI2LD),                               // 无符号整数转化为浮点数
    REP3 (IR_OPCODE, F2I, D2I, LD2I),                                  // 浮点数转化为无符号整数
    REP6 (IR_OPCODE, F2D, F2LD, D2F, D2LD, LD2F, LD2D),                // 浮点数之间相互转化
    REP5 (IR_OPCODE, NEG, NEGS, FNEG, DNEG, LDNEG),                    // 正负符号改变
    /** 
     * 三操作数算术、逻辑、跳转指令（114）
     * 算术：[OP] <operand1>, <operand2>, <operand3>
     * 跳转：[OP] <label>, <operand1>, <operand2>
     */ 
    REP5 (IR_OPCODE, ADD, ADDS, FADD, DADD, LDADD),                    // 加法指令
    REP5 (IR_OPCODE, SUB, SUBS, FSUB, DSUB, LDSUB),                    // 减法指令
    REP5 (IR_OPCODE, MUL, MULS, FMUL, DMUL, LDMUL),                    // 乘法指令
    REP7 (IR_OPCODE, DIV, DIVS, UDIV, UDIVS, FDIV, DDIV, LDDIV),       // 除法指令
    REP4 (IR_OPCODE, MOD, MODS, UMOD, UMODS),                          // 模块操作指令
    REP6 (IR_OPCODE, AND, ANDS, OR, ORS, XOR, XORS),                   // 逻辑运算指令
    REP6 (IR_OPCODE, LSH, LSHS, RSH, RSHS, URSH, URSHS),               // 整数移位指令
    REP5 (IR_OPCODE, EQ, EQS, FEQ, DEQ, LDEQ),                         // 等于判断指令
    REP5 (IR_OPCODE, NE, NES, FNE, DNE, LDNE),                         // 不等判断指令
    REP7 (IR_OPCODE, LT, LTS, ULT, ULTS, FLT, DLT, LDLT),              // 小于判断指令
    REP7 (IR_OPCODE, LE, LES, ULE, ULES, FLE, DLE, LDLE),              // 小于等于判断指令
    REP7 (IR_OPCODE, GT, GTS, UGT, UGTS, FGT, DGT, LDGT),              // 大于判断指令
    REP7 (IR_OPCODE, GE, GES, UGE, UGES, FGE, DGE, LDGE),              // 大于等于判断指令
    REP5 (IR_OPCODE, BEQ, BEQS, FBEQ, DBEQ, LDBEQ),                    // 等于跳转指令
    REP5 (IR_OPCODE, BNE, BNES, FBNE, DBNE, LDBNE),                    // 不等跳转指令
    REP7 (IR_OPCODE, BLT, BLTS, UBLT, UBLTS, FBLT, DBLT, LDBLT),       // 小于跳转指令
    REP7 (IR_OPCODE, BLE, BLES, UBLE, UBLES, FBLE, DBLE, LDBLE),       // 小于等于跳转指令
    REP7 (IR_OPCODE, BGT, BGTS, UBGT, UBGTS, FBGT, DBGT, LDBGT),       // 大于跳转指令
    REP7 (IR_OPCODE, BGE, BGES, UBGE, UBGES, FBGE, DBGE, LDBGE),       // 大于等于跳转指令
    /** 
     * 变长跳转指令（5）
     * 无条件：[OP] <label>
     * 有条件：[OP] <label>, <condition> 
     */
    REP5 (IR_OPCODE, JMP, BT, BTS, BF, BFS),
    /** 函数调用指令（2）：[OP] <prototype>, <ref/addr of fn>, <prototype==void ? result : arguments> ... */
    REP2 (IR_OPCODE, CALL, INLINE),
    /** 多目标跳转指令（1）：[OP] <index>, <label_0>, <label_1>, ... , <label_[index]>, ... */
    REP1 (IR_OPCODE, SWITCH),
    /** 返回指令（1）：[OP] <int> */
    REP1 (IR_OPCODE, RET),
    /** 内存申请指令（1）：[OP] <addr of block entry>, <size> */
    REP1 (IR_OPCODE, ALLOCA),
    /** 块地址获取指令（2）：[OP] <addr of block end> */
    REP2 (IR_OPCODE, BSTART, BEND),
    /** 参数指令（1）：[OP] <addr of va_list>, <size> */
    REP1 (IR_OPCODE, VA_ARG),
    /** 块参数指令（1）：[OP] <addr of va_list>, <size> */
    REP1 (IR_OPCODE, VA_BLOCK_ARG),
    /** 参数定义指令（2） */
    REP1 (IR_OPCODE, VA_START),
    REP1 (IR_OPCODE, VA_END),
    /** 指令标签（1）：对应一个单独的立即数 */
    REP1 (IR_OPCODE, LABEL),
    /** Unspec指令（1）：第一个操作数为 unspec code，其余为参数 */
    REP1 (IR_OPCODE, UNSPEC),
    /** Phi指令（1）：代码生成器内部使用，输出第一个操作数 */
    REP1 (IR_OPCODE, PHI),
    /** 非法指令（1）：指令非法/不可识别 */
    REP1 (IR_OPCODE, INVALID),
    /** 指令边界（1）：不可使用 */
    REP1 (IR_OPCODE, BOUND),
} IROpcode_t;


// IR 操作数为调用类型
static inline int IROpcode_is_call (IROpcode_t code) {
  return code == IR_OPCODE_CALL || code == IR_OPCODE_INLINE;
}
// IR 操作数为整数跳转类型
static inline int IROpcode_is_int_branch (IROpcode_t code) {
  return (code == IR_OPCODE_BT || code == IR_OPCODE_BTS || code == IR_OPCODE_BF || code == IR_OPCODE_BFS || code == IR_OPCODE_BEQ
          || code == IR_OPCODE_BEQS || code == IR_OPCODE_BNE || code == IR_OPCODE_BNES || code == IR_OPCODE_BLT
          || code == IR_OPCODE_BLTS || code == IR_OPCODE_UBLT || code == IR_OPCODE_UBLTS || code == IR_OPCODE_BLE
          || code == IR_OPCODE_BLES || code == IR_OPCODE_UBLE || code == IR_OPCODE_UBLES || code == IR_OPCODE_BGT
          || code == IR_OPCODE_BGTS || code == IR_OPCODE_UBGT || code == IR_OPCODE_UBGTS || code == IR_OPCODE_BGE
          || code == IR_OPCODE_BGES || code == IR_OPCODE_UBGE || code == IR_OPCODE_UBGES);
}
// IR 操作数为浮点数跳转类型
static inline int IROpcode_is_FP_branch (IROpcode_t code) {
  return (code == IR_OPCODE_FBEQ || code == IR_OPCODE_DBEQ || code == IR_OPCODE_LDBEQ || code == IR_OPCODE_FBNE
          || code == IR_OPCODE_DBNE || code == IR_OPCODE_LDBNE || code == IR_OPCODE_FBLT || code == IR_OPCODE_DBLT
          || code == IR_OPCODE_LDBLT || code == IR_OPCODE_FBLE || code == IR_OPCODE_DBLE || code == IR_OPCODE_LDBLE
          || code == IR_OPCODE_FBGT || code == IR_OPCODE_DBGT || code == IR_OPCODE_LDBGT || code == IR_OPCODE_FBGE
          || code == IR_OPCODE_DBGE || code == IR_OPCODE_LDBGE);
}
// IR 操作数为跳转类型
static inline int IROpcode_is_branch (IROpcode_t code) {
  return (code == IR_OPCODE_JMP || IROpcode_is_int_branch (code) || IROpcode_is_FP_branch (code));
}


// ==================================================================================== //
//                                 IR: Data Type
// ==================================================================================== //

/**
 * @brief IR 数据定义类型
 */
typedef enum {
    REP8 (IR_TYPE, I8, U8, I16, U16, I32, U32, I64, U64),   /* Integer types of different size: */
    REP3 (IR_TYPE, F, D, LD),                               /* Float or (long) double type */
    REP2 (IR_TYPE, P, BLK),                                 /* Pointer, memory blocks */
    IR_TYPE (RBLK) = IR_TYPE (BLK) + IR_BLK_NUM,            /* return block */
    REP2 (IR_TYPE, UNDEF, BOUND),
} IRType_t;


static inline int IRType_is_int (IRType_t t) {
    return (IR_TYPE_I8 <= t && t <= IR_TYPE_U64) || t == IR_TYPE_P;
}
static inline int IRType_is_fp (IRType_t t) { 
    return IR_TYPE_F <= t && t <= IR_TYPE_LD; 
}
static inline int IRType_is_blk (IRType_t t) { 
    return IR_TYPE_BLK <= t && t < IR_TYPE_RBLK; 
}
static inline int IRType_is_all_blk (IRType_t t) { 
    return IR_TYPE_BLK <= t && t <= IR_TYPE_RBLK; 
}


// ==================================================================================== //
//                                 IR: Operand
// ==================================================================================== //


/**
 * @brief IR 立即数类型
 */
typedef union {
  int64_t i;
  uint64_t u;
  float f;
  double d;
  long double ld;
} IRImm_t;

/**
 * @brief IR 内存类型
 * @note mem: type[base + index * scale + disp]
 */
typedef struct {
    IRType_t type : 8;
    IRScale_t scale;
    IRReg_t base, index;    /* 0 means no reg for memory.  IR_NON_HARD_REG means no reg for hard reg memory. */
    IRDisp_t disp;
} IRMem_t;

/**
 * @brief IR 字符串类型
 */
typedef struct {
    size_t len;                                             /* 长度 */
    const char *s;                                          /* 地址 */
} IRStr_t;


/**
 * @brief IR 操作数的模式
 */
typedef enum {
    REP8 (IR_OPERAND, UNDEF, REG, HARD_REG, INT, UINT, FLOAT, DOUBLE, LDOUBLE),
    REP6 (IR_OPERAND, REF, STR, MEM, HARD_REG_MEM, LABEL, BOUND),
} IROperandMode_t;


/**
 * @brief IR 指令操作数结构体
 */
typedef struct {
    void *data;                                             /* raw data */
    IROperandMode_t mode;
    IROperandMode_t value_mode;
    union {
        IRReg_t reg;
        IRReg_t hard_reg;                                   /* Used only internally */
        int64_t i;
        uint64_t u;
        float f;
        double d;
        long double ld;
        IRItem_t ref;                                       /* non-export/non-forward after simplification */
        IRStr_t str;
        IRMem_t mem;
        IRMem_t hard_reg_mem;                               /* Used only internally */
        IRLabel_t label;
    } u;
} IROperand_t;


// ==================================================================================== //
//                                 IR: Insn
// ==================================================================================== //

DEF_DLIST_LINK (IRInsn_t);
struct IRInsn {
    void *data;                                             /* raw data */
    DLIST_LINK (IRInsn_t) insn_link;                        /* 指令链表 */
    IROpcode_t code : 32;                                   /* 指令操作码类型 */
    unsigned int nops : 32;                                 /* 指令操作数个数 */
    IROperand_t ops[1];                                     /* 指令操作数类型 */
};
DEF_DLIST (IRInsn_t, insn_link);


// ==================================================================================== //
//                                 IR: Var
// ==================================================================================== //

typedef struct {
    IRType_t type;                                          /* IR_T_BLK .. IR_T_RBLK can be used only args */
    const char *name;
    size_t size;                                            /* ignored for type != [IR_T_BLK .. IR_T_RBLK] */
} IRVar_t;
DEF_VARR (IRVar_t);


typedef struct {
    const char *name;                                       // 函数名称
    IRItem_t func_item;                                     // 函数项
    size_t original_vars_num;                               // 原始变量数量
    DLIST (IRInsn_t) insns, original_insns;                 // 指令列表
    uint32_t nres, nargs, last_temp_num, n_inlines;         // 结果数量、参数数量、最后一个临时变量编号、内联数量
    IRType_t *res_types;                                    // 结果类型数组
    bool vararg_p;                                          // 可变参数标志
    bool expr_p;                                            // 函数是否可以作为链接器表达式的标志
    VARR (IRVar_t) * vars;                                  // 参数和局部变量，但不包括临时变量
    void *machine_code;                                     // 生成的机器码的地址，如果为NULL则表示没有生成机器码
    void *call_addr;                                        // 调用函数的地址，可以与machine_code相同
    void *internal;                                         // 内部数据结构
} *IRFunc_t;


// 函数原型
typedef struct {
    const char *name;                                       // 函数原型名称
    uint32_t nres;                                          // 结果数量
    IRType_t *res_types;                                    // 结果类型数组
    bool vararg_p;                                          // 可变参数标志
    VARR (IRVar_t) * args;                                  // 参数列表，参数名称可以为NULL
} *IRProto_t;  

// 数据
typedef struct {
    const char *name;                                       // 数据名称，可以为NULL
    IRType_t el_type;                                       // 数据定义类型
    size_t nel;                                             // 数据数量
    union {
        long double d;                                      // 用于临时字面量对齐
        uint8_t els[1];                                     // 数据数组
    } u;
} *IRData_t;  

// 引用数据
typedef struct {
    const char *name;                                       // 数据名称，可以为NULL
    IRItem_t ref_item;                                      // 基础项
    int64_t disp;                                           // 相对于基础项的位移
    void *load_addr;
} *IRRefData_t;  

// 表达式数据
typedef struct {
    const char *name;                                       // 表达式名称，可以为NULL
    IRItem_t expr_item;                                     // 在链接期间可以调用的特殊函数
    void *load_addr;
} *IRExprData_t;

// BSS段数据
typedef struct {
    const char *name;                                       // 数据名称，可以为NULL
    uint64_t len;                                           // 长度
} *IRBss_t;


// ==================================================================================== //
//                                 IR: Item
// ==================================================================================== //

// IR 项目类型
typedef enum {
    REP8 (IR_ITEM, func, proto, import, export, forward, data, ref_data, expr_data),
    IR_ITEM (bss),
} IRItemType_t;



DEF_DLIST_LINK (IRItem_t);                                  // 定义一个双向链表链接
struct IRItem {
    void *data;                                             // 数据指针
    IRModule_t module;                                      // 模块类型
    DLIST_LINK (IRItem_t) item_link;                        // 项目链表
    IRItemType_t item_type;                                 // 项目类型
    IRItem_t ref_def;                                       // 引用定义：仅在链接后用于导出/前向项目和导入项目时非空。它形成到最终定义的链
    void *addr;                                             // 调用数据/bss/函数项/定义/原型对象的地址
    bool export_p;                                          // 仅对导出项目为真（仅函数项目）
    bool section_head_p;                                    // 加载后为数据-bss定义。如果它是分配部分的开始，则为true
    union {
        IRFunc_t func;                                      // 函数项
        IRProto_t proto;                                    // 原型对象
        IRName_t import_id;                                 // 导入的ID
        IRName_t export_id;                                 // 导出的ID
        IRName_t forward_id;                                // 前向的ID
        IRData_t data;                                      // 数据项
        IRRefData_t ref_data;                               // 引用的数据项
        IRExprData_t expr_data;                             // 表达式数据项
        IRBss_t bss;                                        // BSS项
    } u;
};
DEF_DLIST (IRItem_t, item_link);                            // 定义一个双向链表


// ==================================================================================== //
//                                 IR: module
// ==================================================================================== //


DEF_DLIST_LINK (IRModule_t);
struct IRModule {
    void *data;
    const char *name;                                       // 模块名
    DLIST (IRItem_t) items;                                 // 模块项
    DLIST_LINK (IRModule_t) module_link;                    // 模块链表
    uint32_t last_temp_item_num;                            // 上个临时项
};
DEF_DLIST (IRModule_t, module_link);



#undef IR_OPCODE
#undef IR_TYPE
#undef IR_OPERAND
#undef IR_ITEM

#endif // EVO_CORE_IR_H