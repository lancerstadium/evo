

#include "ir.h"



// ==================================================================================== //
//                                  Defines
// ==================================================================================== //

#define HARD_REG_NAME_PREFIX "hr"
#define TEMP_REG_NAME_PREFIX "t"
#define TEMP_ITEM_NAME_PREFIX ".lc"
#define OUT_FLAG (1 << 7)

struct gen_ctx;
struct c2mir_ctx;
struct string_ctx;
struct reg_ctx;
struct simplify_ctx;
struct machine_code_ctx;
struct io_ctx;
struct scan_ctx;
struct interp_ctx;



// ==================================================================================== //
//                                 IR: Context
// ==================================================================================== //

DEF_VARR (IRInsn_t);
DEF_VARR (IRReg_t);
DEF_VARR (IROperand_t);
DEF_VARR (IRType_t);
DEF_HTAB (IRItem_t);
DEF_VARR (IRModule_t);
DEF_VARR (size_t);
DEF_VARR (char);
DEF_VARR (uint8_t);
DEF_VARR (IRProto_t);
struct IRContext {
    struct gen_ctx *gen_ctx;     /* should be the 1st member */
    struct c2mir_ctx *c2mir_ctx; /* should be the 2nd member */
    VARR (size_t) * insn_nops;          /* constant after initialization */
    VARR (IRProto_t) * unspec_protos; /* protos of unspec insns (set only during initialization) */
    VARR (char) * temp_string;
    VARR (uint8_t) * temp_data;
    HTAB (IRItem_t) * module_item_tab;
    IRModule_t environment_module;
    IRModule_t curr_module;
    IRFunc_t curr_func;
    int curr_label_num;
    DLIST (IRModule_t) all_modules;
    VARR (IRModule_t) * modules_to_link;
    struct string_ctx *string_ctx;
    struct reg_ctx *reg_ctx;
    struct simplify_ctx *simplify_ctx;
    struct machine_code_ctx *machine_code_ctx;
    struct io_ctx *io_ctx;
    struct scan_ctx *scan_ctx;
    struct interp_ctx *interp_ctx;
    void *setjmp_addr; /* used in interpreter to call setjmp directly not from a shim and FFI */
};



// ==================================================================================== //
//                                 IR: reserved name
// ==================================================================================== //

// 检查是否为临时项目名称
bool _IR_is_reserved_ref_name (IRContext_t ctx, const char *name) {
    return strncmp (name, TEMP_ITEM_NAME_PREFIX, strlen (TEMP_ITEM_NAME_PREFIX)) == 0;
}
// 检查是否为保留名称
bool _IR_is_reserved_name (IRContext_t ctx, const char *name) {
    size_t i, start;
    if (_IR_is_reserved_ref_name (ctx, name))
        return true;
    else if (strncmp (name, HARD_REG_NAME_PREFIX, strlen (HARD_REG_NAME_PREFIX)) == 0)
        start = strlen (HARD_REG_NAME_PREFIX);
    else
        return false;
    for (i = start; name[i] != '\0'; i++)
        if (name[i] < '0' || name[i] > '9') return false;
    return true;
}



// ==================================================================================== //
//                                 IR: Opcode describe
// ==================================================================================== //


struct IROpcode_desc {
    IROpcode_t code;
    const char *name;
    unsigned char op_modes[5];
};

static const struct IROpcode_desc IROpcode_tbl[] = {
  {IR_OPCODE_MOV,           "mov",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FMOV,          "fmov",         {IR_OPERAND_FLOAT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DMOV,          "dmov",         {IR_OPERAND_DOUBLE | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDMOV,         "ldmov",        {IR_OPERAND_LDOUBLE | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_EXT8,          "ext8",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_EXT16,         "ext16",        {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_EXT32,         "ext32",        {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UEXT8,         "uext8",        {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UEXT16,        "uext16",       {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UEXT32,        "uext32",       {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_I2F,           "i2f",          {IR_OPERAND_FLOAT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_I2D,           "i2d",          {IR_OPERAND_DOUBLE | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_I2LD,          "i2ld",         {IR_OPERAND_LDOUBLE | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UI2F,          "ui2f",         {IR_OPERAND_FLOAT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UI2D,          "ui2d",         {IR_OPERAND_DOUBLE | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UI2LD,         "ui2ld",        {IR_OPERAND_LDOUBLE | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_F2I,           "f2i",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_D2I,           "d2i",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LD2I,          "ld2i",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_F2D,           "f2d",          {IR_OPERAND_DOUBLE | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_F2LD,          "f2ld",         {IR_OPERAND_LDOUBLE | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_D2F,           "d2f",          {IR_OPERAND_FLOAT | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_D2LD,          "d2ld",         {IR_OPERAND_LDOUBLE | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LD2F,          "ld2f",         {IR_OPERAND_FLOAT | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LD2D,          "ld2d",         {IR_OPERAND_DOUBLE | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_NEG,           "neg",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_NEGS,          "negs",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FNEG,          "fneg",         {IR_OPERAND_FLOAT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DNEG,          "dneg",         {IR_OPERAND_DOUBLE | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDNEG,         "ldneg",        {IR_OPERAND_LDOUBLE | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_ADD,           "add",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_ADDS,          "adds",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FADD,          "fadd",         {IR_OPERAND_FLOAT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DADD,          "dadd",         {IR_OPERAND_DOUBLE | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDADD,         "ldadd",        {IR_OPERAND_LDOUBLE | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_SUB,           "sub",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_SUBS,          "subs",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FSUB,          "fsub",         {IR_OPERAND_FLOAT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DSUB,          "dsub",         {IR_OPERAND_DOUBLE | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDSUB,         "ldsub",        {IR_OPERAND_LDOUBLE | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_MUL,           "mul",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_MULS,          "muls",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FMUL,          "fmul",         {IR_OPERAND_FLOAT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DMUL,          "dmul",         {IR_OPERAND_DOUBLE | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDMUL,         "ldmul",        {IR_OPERAND_LDOUBLE | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_DIV,           "div",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DIVS,          "divs",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UDIV,          "udiv",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UDIVS,         "udivs",        {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FDIV,          "fdiv",         {IR_OPERAND_FLOAT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DDIV,          "ddiv",         {IR_OPERAND_DOUBLE | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDDIV,         "lddiv",        {IR_OPERAND_LDOUBLE | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_MOD,           "mod",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_MODS,          "mods",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UMOD,          "umod",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UMODS,         "umods",        {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_AND,           "and",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_ANDS,          "ands",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_OR,            "or",           {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_ORS,           "ors",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_XOR,           "xor",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_XORS,          "xors",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_LSH,           "lsh",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_LSHS,          "lshs",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_RSH,           "rsh",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_RSHS,          "rshs",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_URSH,          "ursh",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_URSHS,         "urshs",        {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_EQ,            "eq",           {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_EQS,           "eqs",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FEQ,           "feq",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DEQ,           "deq",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDEQ,          "ldeq",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_NE,            "ne",           {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_NES,           "nes",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FNE,           "fne",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DNE,           "dne",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDNE,          "ldne",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LT,            "lt",           {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_LTS,           "lts",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_ULT,           "ult",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_ULTS,          "ults",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FLT,           "flt",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DLT,           "dlt",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDLT,          "ldlt",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LE,            "le",           {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_LES,           "les",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_ULE,           "ule",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_ULES,          "ules",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FLE,           "fle",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DLE,           "dle",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDLE,          "ldle",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_GT,            "gt",           {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_GTS,           "gts",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UGT,           "ugt",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UGTS,          "ugts",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FGT,           "fgt",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DGT,           "dgt",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDGT,          "ldgt",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_GE,            "ge",           {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_GES,           "ges",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UGE,           "uge",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UGES,          "uges",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FGE,           "fge",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DGE,           "dge",          {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDGE,          "ldge",         {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_JMP,           "jmp",          {IR_OPERAND_LABEL, IR_OPERAND_BOUND}},
  {IR_OPCODE_BT,            "bt",           {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_BTS,           "bts",          {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_BF,            "bf",           {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_BFS,           "bfs",          {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_BEQ,           "beq",          {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_BEQS,          "beqs",         {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FBEQ,          "fbeq",         {IR_OPERAND_LABEL, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DBEQ,          "dbeq",         {IR_OPERAND_LABEL, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDBEQ,         "ldbeq",        {IR_OPERAND_LABEL, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_BNE,           "bne",          {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_BNES,          "bnes",         {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FBNE,          "fbne",         {IR_OPERAND_LABEL, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DBNE,          "dbne",         {IR_OPERAND_LABEL, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDBNE,         "ldbne",        {IR_OPERAND_LABEL, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_BLT,           "blt",          {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_BLTS,          "blts",         {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UBLT,          "ublt",         {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UBLTS,         "ublts",        {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FBLT,          "fblt",         {IR_OPERAND_LABEL, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DBLT,          "dblt",         {IR_OPERAND_LABEL, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDBLT,         "ldblt",        {IR_OPERAND_LABEL, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_BLE,           "ble",          {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_BLES,          "bles",         {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UBLE,          "uble",         {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UBLES,         "ubles",        {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FBLE,          "fble",         {IR_OPERAND_LABEL, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DBLE,          "dble",         {IR_OPERAND_LABEL, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDBLE,         "ldble",        {IR_OPERAND_LABEL, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_BGT,           "bgt",          {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_BGTS,          "bgts",         {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UBGT,          "ubgt",         {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UBGTS,         "ubgts",        {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FBGT,          "fbgt",         {IR_OPERAND_LABEL, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DBGT,          "dbgt",         {IR_OPERAND_LABEL, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDBGT,         "ldbgt",        {IR_OPERAND_LABEL, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_BGE,           "bge",          {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_BGES,          "bges",         {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UBGE,          "ubge",         {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_UBGES,         "ubges",        {IR_OPERAND_LABEL, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_FBGE,          "fbge",         {IR_OPERAND_LABEL, IR_OPERAND_FLOAT, IR_OPERAND_FLOAT, IR_OPERAND_BOUND}},
  {IR_OPCODE_DBGE,          "dbge",         {IR_OPERAND_LABEL, IR_OPERAND_DOUBLE, IR_OPERAND_DOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_LDBGE,         "ldbge",        {IR_OPERAND_LABEL, IR_OPERAND_LDOUBLE, IR_OPERAND_LDOUBLE, IR_OPERAND_BOUND}},
  {IR_OPCODE_CALL,          "call",         {IR_OPERAND_BOUND}},
  {IR_OPCODE_INLINE,        "inline",       {IR_OPERAND_BOUND}},
  {IR_OPCODE_SWITCH,        "switch",       {IR_OPERAND_BOUND}},
  {IR_OPCODE_RET,           "ret",          {IR_OPERAND_BOUND}},
  {IR_OPCODE_ALLOCA,        "alloca",       {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_BSTART,        "bstart",       {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_BOUND}},
  {IR_OPCODE_BEND,          "bend",         {IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_VA_ARG,        "va_arg",       {IR_OPERAND_INT | OUT_FLAG, IR_OPERAND_INT, IR_OPERAND_UNDEF, IR_OPERAND_BOUND}},
  {IR_OPCODE_VA_BLOCK_ARG,  "va_block_arg", {IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_VA_START,      "va_start",     {IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_VA_END,        "va_end",       {IR_OPERAND_INT, IR_OPERAND_BOUND}},
  {IR_OPCODE_LABEL,         "label",        {IR_OPERAND_BOUND}},
  {IR_OPCODE_UNSPEC,        "unspec",       {IR_OPERAND_BOUND}},
  {IR_OPCODE_PHI,           "phi",          {IR_OPERAND_BOUND}},
  {IR_OPCODE_INVALID,       "invalid",      {IR_OPERAND_BOUND}},
};



static void check_opcode_tbl(IRContext_t ctx) {
    size_t i, j;
    VARR_CREATE (size_t, ctx->insn_nops, 0);
    for (i = 0; i < IR_OPCODE_BOUND; i++) {
        log_assert (IROpcode_tbl[i].code == i);
        for (j = 0; IROpcode_tbl[i].op_modes[j] != IR_OPERAND_BOUND; j++)
        ;
        VARR_PUSH (size_t, ctx->insn_nops, j);
    }
}

static IROperandMode_t type2mode(IRType_t type) {
  return (type == IR_TYPE_UNDEF ? IR_OPERAND_UNDEF
          : type == IR_TYPE_F   ? IR_OPERAND_FLOAT
          : type == IR_TYPE_D   ? IR_OPERAND_DOUBLE
          : type == IR_TYPE_LD  ? IR_OPERAND_LDOUBLE
                                : IR_OPERAND_INT);
}



// ==================================================================================== //
//                                 IR: String
// ==================================================================================== //


typedef struct {
    size_t num;               /* string number starting with 1 */
    IRStr_t str;
} string_t;


DEF_VARR (string_t);
DEF_HTAB (string_t);
struct string_ctx {
    VARR (string_t) * strings;
    HTAB (string_t) * string_tab;
};


static htab_hash_t str_hash(string_t str, void *arg) {
    return mir_hash(str.str.s, str.str.len, 0);
}
static int str_eq(string_t str1, string_t str2, void *arg) {
    return str1.str.len == str2.str.len && memcmp(str1.str.s, str2.str.s, str1.str.len) == 0;
}
static void string_init(VARR(string_t) **strs, HTAB(string_t) **str_tab) {
    string_t string = {0, {0, NULL}};
    VARR_CREATE (string_t, *strs, 0);
    VARR_PUSH (string_t, *strs, string); /* don't use 0th string */
    HTAB_CREATE (string_t, *str_tab, 1000, str_hash, str_eq, NULL);
}
static int string_find(VARR (string_t) * *strs, HTAB (string_t) * *str_tab, IRStr_t str, string_t *s) {
    string_t string;
    string.str = str;
    return HTAB_DO (string_t, *str_tab, string, HTAB_FIND, *s);
}
static string_t string_store(IRContext_t ctx, VARR (string_t) * *strs, HTAB (string_t) * *str_tab, IRStr_t str) {
    char *heap_str;
    string_t el, string;
    if (string_find (strs, str_tab, str, &el)) return el;
    if ((heap_str = malloc (str.len)) == NULL)
        log_error("Not enough memory for strings");
    memcpy (heap_str, str.s, str.len);
    string.str.s = heap_str;
    string.str.len = str.len;
    string.num = VARR_LENGTH (string_t, *strs);
    VARR_PUSH (string_t, *strs, string);
    HTAB_DO (string_t, *str_tab, string, HTAB_INSERT, el);
    return string;
}
static string_t get_ctx_string(IRContext_t ctx, IRStr_t str) {
    return string_store (ctx, &ctx->string_ctx->strings, &ctx->string_ctx->string_tab, str);
}
static const char *get_ctx_str(IRContext_t ctx, const char *string) {
    return get_ctx_string (ctx, (IRStr_t){strlen (string) + 1, string}).str.s;
}
static void string_finish(VARR (string_t) * *strs, HTAB (string_t) * *str_tab) {
    size_t i;
    for (i = 1; i < VARR_LENGTH (string_t, *strs); i++)
        free ((char *) VARR_ADDR (string_t, *strs)[i].str.s);
    VARR_DESTROY (string_t, *strs);
    HTAB_DESTROY (string_t, *str_tab);
}




// ==================================================================================== //
//                                 IR: Reg describe
// ==================================================================================== //


typedef struct reg_desc {
    char *name;                     /* 1st key for the name2rdn hash tab */
    IRType_t type;
    IRReg_t reg;                    /* 1st key reg2rdn hash tab */
} reg_desc_t;

DEF_VARR (reg_desc_t);
DEF_HTAB (size_t);
typedef struct func_regs {
    VARR (reg_desc_t) * reg_descs;
    HTAB (size_t) * name2rdn_tab;
    HTAB (size_t) * reg2rdn_tab;
} *func_regs_t;

static int name2rdn_eq (size_t rdn1, size_t rdn2, void *arg) {
    func_regs_t func_regs = arg;
    reg_desc_t *addr = VARR_ADDR (reg_desc_t, func_regs->reg_descs);
    return strcmp (addr[rdn1].name, addr[rdn2].name) == 0;
}

static htab_hash_t name2rdn_hash (size_t rdn, void *arg) {
    func_regs_t func_regs = arg;
    reg_desc_t *addr = VARR_ADDR (reg_desc_t, func_regs->reg_descs);
    return mir_hash (addr[rdn].name, strlen (addr[rdn].name), 0);
}

static int reg2rdn_eq (size_t rdn1, size_t rdn2, void *arg) {
    func_regs_t func_regs = arg;
    reg_desc_t *addr = VARR_ADDR (reg_desc_t, func_regs->reg_descs);
    return addr[rdn1].reg == addr[rdn2].reg;
}

static htab_hash_t reg2rdn_hash (size_t rdn, void *arg) {
    func_regs_t func_regs = arg;
    reg_desc_t *addr = VARR_ADDR (reg_desc_t, func_regs->reg_descs);
    return mir_hash_finish (mir_hash_step (mir_hash_init (0), addr[rdn].reg));
}

static void func_regs_init (IRContext_t ctx, IRFunc_t func) {
    func_regs_t func_regs;
    reg_desc_t rd = {NULL, IR_TYPE_I64, 0};
    if ((func_regs = func->internal = malloc (sizeof (struct func_regs))) == NULL)
        log_error("Not enough memory for func regs info");
    VARR_CREATE (reg_desc_t, func_regs->reg_descs, 50);
    VARR_PUSH (reg_desc_t, func_regs->reg_descs, rd); /* for 0 reg */
    HTAB_CREATE (size_t, func_regs->name2rdn_tab, 100, name2rdn_hash, name2rdn_eq, func_regs);
    HTAB_CREATE (size_t, func_regs->reg2rdn_tab, 100, reg2rdn_hash, reg2rdn_eq, func_regs);
}

static IRReg_t create_func_reg (IRContext_t ctx, IRFunc_t func, const char *name, IRReg_t reg, IRType_t type, int any_p, char **name_ptr) {
    func_regs_t func_regs = func->internal;
    reg_desc_t rd;
    size_t rdn, tab_rdn;
    int htab_res;

    if (!any_p && _IR_is_reserved_name (ctx, name))
        log_error("redefining a reserved name %s", name);
    rd.name = (char *) name;
    rd.type = type;
    rd.reg = reg; /* 0 is reserved */
    rdn = VARR_LENGTH (reg_desc_t, func_regs->reg_descs);
    VARR_PUSH (reg_desc_t, func_regs->reg_descs, rd);
    if (HTAB_DO (size_t, func_regs->name2rdn_tab, rdn, HTAB_FIND, tab_rdn)) {
        VARR_POP (reg_desc_t, func_regs->reg_descs);
        log_error("Repeated reg declaration %s", name);
    }
    if ((rd.name = malloc (strlen (name) + 1)) == NULL)
        log_error("Not enough memory for reg names");
    VARR_ADDR (reg_desc_t, func_regs->reg_descs)[rdn].name = *name_ptr = rd.name;
    strcpy (*name_ptr, name);
    htab_res = HTAB_DO (size_t, func_regs->name2rdn_tab, rdn, HTAB_INSERT, tab_rdn);
    log_assert (!htab_res);
    htab_res = HTAB_DO (size_t, func_regs->reg2rdn_tab, rdn, HTAB_INSERT, tab_rdn);
    log_assert (!htab_res);
    return reg;
}

static void func_regs_finish (IRContext_t ctx, IRFunc_t func) {
    func_regs_t func_regs = func->internal;
    char *name;
    for (size_t i = 0; i < VARR_LENGTH (reg_desc_t, func_regs->reg_descs); i++)
        if ((name = VARR_GET (reg_desc_t, func_regs->reg_descs, i).name) != NULL) free (name);
    VARR_DESTROY (reg_desc_t, func_regs->reg_descs);
    HTAB_DESTROY (size_t, func_regs->name2rdn_tab);
    HTAB_DESTROY (size_t, func_regs->reg2rdn_tab);
    free (func->internal);
    func->internal = NULL;
}



// ==================================================================================== //
//                                 IR: Opcode describe
// ==================================================================================== //
