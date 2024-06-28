
#include <isa/rv/def.h>
#include <evo/task.h>

// ==================================================================================== //
//                                    rv: Reg                                      
// ==================================================================================== //


RegDef_fn_def(RV);


// ==================================================================================== //
//                                    rv: Insn                                      
// ==================================================================================== //

InsnDef_fn_def(RV);

Insn_fn_def(RV);

// ==================================================================================== //
//                                    rv: CPUState                                      
// ==================================================================================== //

CPUState_fn_def(RV);

Val* CPUState_OP_def(RV, fetch)(CPUState(RV) * cpu) {
    Val* res = CPUState_get_mem(RV, cpu, cpu->snpc, 4);
    Val_inc(cpu->snpc, res->len);
    return res;
}

Insn(RV) * CPUState_OP_def(RV, decode)(CPUState(RV) * cpu, Val * val) {
    Val_copy(cpu->dnpc, cpu->snpc);
    Insn(RV) * insn = Insn_decode(RV, val);
    return insn;
}

#define RV_EXEC_R(...)                             \
    do {                                           \
        u8 rd = Val_as_u8(insn->oprs[0], 0);       \
        u8 r1 = Val_as_u8(insn->oprs[1], 0);       \
        u8 r2 = Val_as_u8(insn->oprs[2], 0);       \
        Val* r1_v = CPUState_get_reg(RV, cpu, r1); \
        Val* r2_v = CPUState_get_reg(RV, cpu, r2); \
        Val* res = Val_new_i64(__VA_ARGS__);       \
        CPUState_set_reg(RV, cpu, rd, res);        \
        Val_free(r1_v);                            \
        Val_free(r2_v);                            \
        Val_free(res);                             \
    } while (0)

#define RV_EXEC_I(...)                             \
    do {                                           \
        u8 rd = Val_as_u8(insn->oprs[0], 0);       \
        u8 r1 = Val_as_u8(insn->oprs[1], 0);       \
        i64 immi = Val_as_i64(insn->oprs[2], 0);   \
        Val* r1_v = CPUState_get_reg(RV, cpu, r1); \
        Val* res = Val_new_i64(__VA_ARGS__);       \
        CPUState_set_reg(RV, cpu, rd, res);        \
        Val_free(r1_v);                            \
        Val_free(res);                             \
    } while (0)

#define RV_EXEC_I_M(S, A, L)                                        \
    do {                                                            \
        u8 rd = Val_as_u8(insn->oprs[0], 0);                        \
        u8 r1 = Val_as_u8(insn->oprs[1], 0);                        \
        i64 immi = Val_as_i64(insn->oprs[2], 0);                    \
        Val* r1_v = CPUState_get_reg(RV, cpu, r1);                  \
        Val* res = Val_to_##S##64(CPUState_get_mem(RV, cpu, A, L)); \
        CPUState_set_reg(RV, cpu, rd, res);                         \
        Val_free(r1_v);                                             \
        Val_free(res);                                              \
    } while (0)

#define RV_EXEC_U(...)                           \
    do {                                         \
        u8 rd = Val_as_u8(insn->oprs[0], 0);     \
        i64 immu = Val_as_i64(insn->oprs[1], 0); \
        Val* res = Val_new_i64(__VA_ARGS__);     \
        CPUState_set_reg(RV, cpu, rd, res);      \
        Val_free(res);                           \
    } while (0)

#define RV_EXEC_S_M(N)                                                               \
    do {                                                                             \
        u8 r1 = Val_as_u8(insn->oprs[0], 0);                                         \
        u8 r2 = Val_as_u8(insn->oprs[1], 0);                                         \
        i64 imms = Val_as_i64(insn->oprs[2], 0);                                     \
        Val* r1_v = CPUState_get_reg(RV, cpu, r1);                                   \
        Val* r2_v = CPUState_get_reg(RV, cpu, r2);                                   \
        CPUState_set_mem(RV, cpu, Val_new_u64(Val_as_i64(r1_v, 0) + imms), r2_v, N); \
        Val_free(r1_v);                                                              \
        Val_free(r2_v);                                                              \
    } while (0)

void CPUState_OP_def(RV, execute)(CPUState(RV) * cpu, Insn(RV) * insn) {
    switch(insn->id) {
        /* RV32I: R-Type Arithmetic */
        case RV_ADD     : RV_EXEC_R(Val_as_i64(r1_v, 0)  +  Val_as_i64(r2_v, 0));       break;
        case RV_SUB     : RV_EXEC_R(Val_as_i64(r1_v, 0)  -  Val_as_i64(r2_v, 0));       break;
        case RV_XOR     : RV_EXEC_R(Val_as_i64(r1_v, 0)  ^  Val_as_i64(r2_v, 0));       break;
        case RV_OR      : RV_EXEC_R(Val_as_i64(r1_v, 0)  |  Val_as_i64(r2_v, 0));       break;
        case RV_AND     : RV_EXEC_R(Val_as_i64(r1_v, 0)  &  Val_as_i64(r2_v, 0));       break;
        case RV_SLL     : RV_EXEC_R(Val_as_u64(r1_v, 0) <<  Val_as_i64(r2_v, 0));       break;
        case RV_SRL     : RV_EXEC_R(Val_as_u64(r1_v, 0) >>  Val_as_i64(r2_v, 0));       break;
        case RV_SRA     : RV_EXEC_R(Val_as_i64(r1_v, 0) >>  Val_as_i64(r2_v, 0));       break;
        case RV_SLT     : RV_EXEC_R(Val_as_i64(r1_v, 0) <   Val_as_i64(r2_v, 0));       break;
        case RV_SLTU    : RV_EXEC_R(Val_as_u64(r1_v, 0) <   Val_as_u64(r2_v, 0));       break;
        /* RV32I: I-Type Arithmetic */
        case RV_ADDI    : RV_EXEC_I(Val_as_i64(r1_v, 0)  +  immi);                      break;
        case RV_XORI    : RV_EXEC_I(Val_as_i64(r1_v, 0)  ^  immi);                      break;
        case RV_ORI     : RV_EXEC_I(Val_as_i64(r1_v, 0)  |  immi);                      break;
        case RV_ANDI    : RV_EXEC_I(Val_as_i64(r1_v, 0)  &  immi);                      break;
        case RV_SLLI    : RV_EXEC_I(Val_as_i64(r1_v, 0) <<  immi);                      break;
        case RV_SRLI    : RV_EXEC_I(Val_as_i64(r1_v, 0) >>  immi);                      break;
        case RV_SRAI    : RV_EXEC_I(Val_as_i64(r1_v, 0) >>  immi);                      break;
        case RV_SLTI    : RV_EXEC_I(Val_as_i64(r1_v, 0)  <  immi);                      break;
        case RV_SLTIU   : RV_EXEC_I(Val_as_u64(r1_v, 0)  <  (u64)immi);                 break;
        /* RV32I: U-Type Arithmetic */
        case RV_LUI     : RV_EXEC_U(immu << 12);                                        break;
        case RV_AUIPC   : RV_EXEC_U(Val_as_u64(cpu->pc, 0) + immu);                     break;
        /* RV32I: Load I-Type */
        case RV_LB      : RV_EXEC_I_M(u, Val_new_u64(Val_as_u64(r1_v, 0) + immi), 1);   break;
        case RV_LH      : RV_EXEC_I_M(u, Val_new_u64(Val_as_u64(r1_v, 0) + immi), 2);   break;
        case RV_LW      : RV_EXEC_I_M(u, Val_new_u64(Val_as_u64(r1_v, 0) + immi), 4);   break;
        case RV_LBU     : RV_EXEC_I_M(i, Val_new_u64(Val_as_u64(r1_v, 0) + immi), 1);   break;
        case RV_LHU     : RV_EXEC_I_M(i, Val_new_u64(Val_as_u64(r1_v, 0) + immi), 2);   break;
        /* RV32I: Store S-Type */
        case RV_SB      : RV_EXEC_S_M(8);                                               break;
        case RV_SH      : RV_EXEC_S_M(16);                                              break;
        case RV_SW      : RV_EXEC_S_M(32);                                              break;
        /* RV32I: Branch */

        default: break;
    }
    Val_copy(cpu->pc, cpu->dnpc);
}

// ==================================================================================== //
//                                    rv: Task                                     
// ==================================================================================== //


Insn(RV) * TaskCtx_OP_ISA_def(Exec, run, RV) (UNUSED TaskCtx(Exec) *ctx, UNUSED Val* bc) {
    // Match 
    Insn(RV) * insn = Insn_match(RV, bc);
    if(insn != NULL) {
        UNUSED InsnDef(RV) * def = INSN(RV, insn->id);
    }
    return insn;
}