
#include <isa/rv/def.h>

// ==================================================================================== //
//                                    rv: Reg                                      
// ==================================================================================== //


RegDef_fn_def(RV);


// ==================================================================================== //
//                                    rv: Insn                                      
// ==================================================================================== //

InsnDef_fn_def(RV);

Insn_fn_def(RV);


Insn(RV) * Insn_OP_def(RV, encode)(Insn(RV) * insn, Val * args[]) {
    Log_ast(insn != NULL, "Insn: insn is null");
    Log_ast(args != NULL, "Insn: args are null");
    InsnDef(RV)* df = INSN(RV, insn->id);
    for (size_t i = 0; i < insn->len; i++) {
        if (args[i] != NULL) {
            BitMap* bm = (df->tr.t[i]).map;
            size_t bml = (df->tr.t[i]).len;
            insn->oprs[i] = Val_tymatch(args[i], &df->tr.t[i]);
            Val_imp_map(&insn->bc, bm, bml, args[i]);
        }
    }
    return insn;
}
Insn(RV) * Insn_OP_def(RV, decode)(Val* bc) {
    Insn(RV)* insn = Insn_OP(RV, match)(bc);
    if (insn != NULL) {
        InsnDef(RV)* df = INSN(RV, insn->id);
        for (size_t i = 0; i < insn->len; i++) {
            BitMap* bm = (df->tr.t[i]).map;
            size_t bml = (df->tr.t[i]).len;
            insn->oprs[i] = Val_ext_map(bc, bm, bml);
            Val_copy(insn->bc, bc);
        }
    }
    return insn;
}


static int rv_decode4(u32 inst, Insn(RV)* restrict rv_inst) {
    enum {
        ENC_F_RD  = 1 << 0,
        ENC_F_RS1 = 1 << 1,
        ENC_F_RS2 = 1 << 2,
        ENC_F_RS3 = 1 << 3,
        ENC_F_IMM_MASK = 7 << 4, // 3 bits
        ENC_F_IMM_U = 1 << 4,
        ENC_F_IMM_I = 2 << 4,
        ENC_F_IMM_S = 3 << 4,
        ENC_F_IMM_J = 4 << 4,
        ENC_F_IMM_B = 5 << 4,
        ENC_F_IMM_SHAMT = 6 << 4,
        ENC_F_IMM_AMO = 7 << 4,
        ENC_F_RM  = 1 << 7, // rounding mode

        ENC_R = ENC_F_RD | ENC_F_RS1 | ENC_F_RS2,
        ENC_R2 = ENC_F_RD | ENC_F_RS1,
        ENC_R4 = ENC_F_RD | ENC_F_RS1 | ENC_F_RS2 | ENC_F_RS3,
        ENC_I = ENC_F_RD | ENC_F_RS1 | ENC_F_IMM_I,
        ENC_I_SHAMT = ENC_F_RD | ENC_F_RS1 | ENC_F_IMM_SHAMT,
        ENC_S = ENC_F_RS1 | ENC_F_RS2 | ENC_F_IMM_S,
        ENC_B = ENC_F_RS1 | ENC_F_RS2 | ENC_F_IMM_B,
        ENC_U = ENC_F_RD | ENC_F_IMM_U,
        ENC_J = ENC_F_RD | ENC_F_IMM_J,
    };

    unsigned opcode = UBFX(inst, 2, 6);
    unsigned funct3 = UBFX(inst, 12, 14);
    unsigned funct7 = UBFX(inst, 25, 31);
    unsigned mnem = 0, encoding = 0;
    switch (opcode) {
    case 0x00: encoding = ENC_I;
        mnem = (const u16[]) {RV_LB, RV_LH, RV_LW, RV_LD, RV_LBU, RV_LHU, RV_LWU, 0}[funct3];
        break;
    case 0x01: encoding = ENC_I;
        mnem = (const u16[]) {0, 0, RV_FLW, RV_FLD, 0, 0, 0, 0}[funct3];
        break;
    case 0x03: encoding = ENC_I;
        mnem = (const u16[]) {RV_FENCE, RV_FENCEI, 0, 0, 0, 0, 0, 0}[funct3];
        break;
    case 0x04:
        switch (funct3) {
        case 0: encoding = ENC_I; mnem = RV_ADDI; break;
        case 1: encoding = ENC_I_SHAMT; mnem = RV_SLLI; break;
        case 2: encoding = ENC_I; mnem = RV_SLTI; break;
        case 3: encoding = ENC_I; mnem = RV_SLTIU; break;
        case 4: encoding = ENC_I; mnem = RV_XORI; break;
        case 5: encoding = ENC_I_SHAMT; mnem = funct7 & 0x20 ? RV_SRAI : RV_SRLI; break;
        case 6: encoding = ENC_I; mnem = RV_ORI; break;
        case 7: encoding = ENC_I; mnem = RV_ANDI; break;
        }
        break;
    case 0x05: encoding = ENC_U; mnem = RV_AUIPC; break;
    case 0x06:
        switch (funct3) {
        case 0: encoding = ENC_I; mnem = RV_ADDIW; break;
        case 1: encoding = ENC_I_SHAMT; mnem = RV_SLLIW; break;
        case 5: encoding = ENC_I_SHAMT; mnem = funct7 & 0x20 ? RV_SRAIW : RV_SRLIW; break;
        default: return RV_UNDEF;
        }
        break;
    case 0x08: encoding = ENC_S;
        mnem = (const u16[]) {RV_SB, RV_SH, RV_SW, RV_SD, 0, 0, 0, 0}[funct3];
        break;
    case 0x09: encoding = ENC_S;
        mnem = (const u16[]) {0, 0, RV_FSW, RV_FSD, 0, 0, 0, 0}[funct3];
        break;
    case 0x0b: encoding = ENC_R | ENC_F_IMM_AMO;
        switch (funct7 >> 2) {
        case 0x00: mnem = (const u16[]) {0, 0, RV_AMOADDW, RV_AMOADDD, 0, 0, 0, 0}[funct3]; break;
        case 0x01: mnem = (const u16[]) {0, 0, RV_AMOSWAPW, RV_AMOSWAPD, 0, 0, 0, 0}[funct3]; break;
        case 0x02: mnem = (const u16[]) {0, 0, RV_LRW, RV_LRD, 0, 0, 0, 0}[funct3]; break;
        case 0x03: mnem = (const u16[]) {0, 0, RV_SCW, RV_SCD, 0, 0, 0, 0}[funct3]; break;
        case 0x04: mnem = (const u16[]) {0, 0, RV_AMOXORW, RV_AMOXORD, 0, 0, 0, 0}[funct3]; break;
        case 0x08: mnem = (const u16[]) {0, 0, RV_AMOORW, RV_AMOORD, 0, 0, 0, 0}[funct3]; break;
        case 0x0c: mnem = (const u16[]) {0, 0, RV_AMOANDW, RV_AMOANDD, 0, 0, 0, 0}[funct3]; break;
        case 0x10: mnem = (const u16[]) {0, 0, RV_AMOMINW, RV_AMOMIND, 0, 0, 0, 0}[funct3]; break;
        case 0x14: mnem = (const u16[]) {0, 0, RV_AMOMAXW, RV_AMOMAXD, 0, 0, 0, 0}[funct3]; break;
        case 0x18: mnem = (const u16[]) {0, 0, RV_AMOMINUW, RV_AMOMINUD, 0, 0, 0, 0}[funct3]; break;
        case 0x1c: mnem = (const u16[]) {0, 0, RV_AMOMAXUW, RV_AMOMAXUD, 0, 0, 0, 0}[funct3]; break;
        default: return RV_UNDEF;
        }
        break;
    case 0x0c: encoding = ENC_R;
        switch (funct7) {
        case 0x00: mnem = (const u16[]) {RV_ADD, RV_SLL, RV_SLT, RV_SLTU, RV_XOR, RV_SRL, RV_OR, RV_AND}[funct3]; break;
        case 0x01: mnem = (const u16[]) {RV_MUL, RV_MULH, RV_MULHSU, RV_MULHU, RV_DIV, RV_DIVU, RV_REM, RV_REMU}[funct3]; break;
        case 0x20: mnem = (const u16[]) {RV_SUB, 0, 0, 0, 0, RV_SRA, 0, 0}[funct3]; break;
        default: return RV_UNDEF;
        }
        break;
    case 0x0d: encoding = ENC_U; mnem = RV_LUI; break;
    case 0x0e: encoding = ENC_R;
        switch (funct7) {
        case 0x00: mnem = (const u16[]) {RV_ADDW, RV_SLLW, 0, 0, 0, RV_SRLW, 0, 0}[funct3]; break;
        case 0x01: mnem = (const u16[]) {RV_MULW, 0, 0, 0, RV_DIVW, RV_DIVUW, RV_REMW, RV_REMUW}[funct3]; break;
        case 0x20: mnem = (const u16[]) {RV_SUBW, 0, 0, 0, 0, RV_SRAW, 0, 0}[funct3]; break;
        default: return RV_UNDEF;
        }
        break;
    case 0x10: encoding = ENC_R4 | ENC_F_RM; mnem = (const u16[4]) {RV_FMADDS, RV_FMADDD}[UBFX(inst, 25, 26)]; break;
    case 0x11: encoding = ENC_R4 | ENC_F_RM; mnem = (const u16[4]) {RV_FMSUBS, RV_FMSUBD}[UBFX(inst, 25, 26)]; break;
    case 0x12: encoding = ENC_R4 | ENC_F_RM; mnem = (const u16[4]) {RV_FNMSUBS, RV_FNMSUBD}[UBFX(inst, 25, 26)]; break;
    case 0x13: encoding = ENC_R4 | ENC_F_RM; mnem = (const u16[4]) {RV_FNMADDS, RV_FNMADDD}[UBFX(inst, 25, 26)]; break;
    case 0x14:
        switch (funct7) {
            case 0x00: encoding = ENC_R | ENC_F_RM; mnem = RV_FADDS; break;
            case 0x01: encoding = ENC_R | ENC_F_RM; mnem = RV_FADDD; break;
            case 0x04: encoding = ENC_R | ENC_F_RM; mnem = RV_FSUBS; break;
            case 0x05: encoding = ENC_R | ENC_F_RM; mnem = RV_FSUBD; break;
            case 0x08: encoding = ENC_R | ENC_F_RM; mnem = RV_FMULS; break;
            case 0x09: encoding = ENC_R | ENC_F_RM; mnem = RV_FMULD; break;
            case 0x0c: encoding = ENC_R | ENC_F_RM; mnem = RV_FDIVS; break;
            case 0x0d: encoding = ENC_R | ENC_F_RM; mnem = RV_FDIVD; break;
            case 0x20: encoding = ENC_R2 | ENC_F_RM; mnem = (const u16[32]) {0, RV_FCVTSD}[UBFX(inst, 20, 24)]; break;
            case 0x21: encoding = ENC_R2 | ENC_F_RM; mnem = (const u16[32]) {RV_FCVTDS}[UBFX(inst, 20, 24)]; break;
            case 0x2c: encoding = ENC_R2 | ENC_F_RM; mnem = RV_FSQRTS; break; // TODO: check rs2
            case 0x2d: encoding = ENC_R2 | ENC_F_RM; mnem = RV_FSQRTD; break; // TODO: check rs2
            case 0x10: encoding = ENC_R; mnem = (const u16[]) {RV_FSGNJS, RV_FSGNJNS, RV_FSGNJXS, 0, 0, 0, 0, 0}[funct3]; break;
            case 0x11: encoding = ENC_R; mnem = (const u16[]) {RV_FSGNJD, RV_FSGNJND, RV_FSGNJXD, 0, 0, 0, 0, 0}[funct3]; break;
            case 0x14: encoding = ENC_R; mnem = (const u16[]) {RV_FMINS, RV_FMAXS, 0, 0, 0, 0, 0, 0}[funct3]; break;
            case 0x15: encoding = ENC_R; mnem = (const u16[]) {RV_FMIND, RV_FMAXD, 0, 0, 0, 0, 0, 0}[funct3]; break;
            case 0x50: encoding = ENC_R; mnem = (const u16[]) {RV_FLES, RV_FLTS, RV_FEQS, 0, 0, 0, 0, 0}[funct3]; break;
            case 0x51: encoding = ENC_R; mnem = (const u16[]) {RV_FLED, RV_FLTD, RV_FEQD, 0, 0, 0, 0, 0}[funct3]; break;
            case 0x60: encoding = ENC_R2 | ENC_F_RM; mnem = (const u16[32]) {RV_FCVTWS, RV_FCVTWUS, RV_FCVTLS, RV_FCVTLUS}[UBFX(inst, 20, 24)]; break;
            case 0x61: encoding = ENC_R2 | ENC_F_RM; mnem = (const u16[32]) {RV_FCVTWD, RV_FCVTWUD, RV_FCVTLD, RV_FCVTLUD}[UBFX(inst, 20, 24)]; break;
            case 0x68: encoding = ENC_R2 | ENC_F_RM; mnem = (const u16[32]) {RV_FCVTSW, RV_FCVTSWU, RV_FCVTSL, RV_FCVTSLU}[UBFX(inst, 20, 24)]; break;
            case 0x69: encoding = ENC_R2 | ENC_F_RM; mnem = (const u16[32]) {RV_FCVTDW, RV_FCVTDWU, RV_FCVTDL, RV_FCVTDLU}[UBFX(inst, 20, 24)]; break;
            case 0x70: encoding = ENC_R2; mnem = (const u16[]) {RV_FMVXW, RV_FCLASSS, 0, 0, 0, 0, 0, 0}[funct3]; break; // TODO: check rs2
            case 0x71: encoding = ENC_R2; mnem = (const u16[]) {RV_FMVXD, RV_FCLASSD, 0, 0, 0, 0, 0, 0}[funct3]; break; // TODO: check rs2
            case 0x78: encoding = ENC_R2; mnem = (const u16[]) {RV_FMVWX, 0, 0, 0, 0, 0, 0, 0}[funct3]; break; // TODO: check rs2
            case 0x79: encoding = ENC_R2; mnem = (const u16[]) {RV_FMVDX, 0, 0, 0, 0, 0, 0, 0}[funct3]; break; // TODO: check rs2
        default: return RV_UNDEF;
        }
        break;
    case 0x18: encoding = ENC_B;
        mnem = (const u16[]) {RV_BEQ, RV_BNE, 0, 0, RV_BLT, RV_BGE, RV_BLTU, RV_BGEU}[funct3];
        break;
    case 0x19: encoding = ENC_I; mnem = RV_JALR; break; // TODO: check funct3
    case 0x1b: encoding = ENC_J; mnem = RV_JAL; break;
    case 0x1c: encoding = ENC_I;
        mnem = (const u16[]) {RV_ECALL, RV_CSRRW, RV_CSRRS, RV_CSRRC, 0, RV_CSRRWI, RV_CSRRSI, RV_CSRRCI}[funct3];
        break;
    }

    if (!mnem)
        return RV_UNDEF;
    rv_inst->id = mnem;
    rv_inst->rd = (encoding & ENC_F_RD) ? UBFX(inst, 7, 11) : RV_REG_INV;
    rv_inst->rs1 = (encoding & ENC_F_RS1) ? UBFX(inst, 15, 19) : RV_REG_INV;
    rv_inst->rs2 = (encoding & ENC_F_RS2) ? UBFX(inst, 20, 24) : RV_REG_INV;
    rv_inst->rs3 = (encoding & ENC_F_RS3) ? UBFX(inst, 27, 31) : RV_REG_INV;
    if (encoding & ENC_F_RM)
        rv_inst->misc = funct3;
    switch (encoding & ENC_F_IMM_MASK) {
    default: rv_inst->imm = 0; break;
    case ENC_F_IMM_U: rv_inst->imm = UBFX(inst, 12, 31) << 12; break;
    case ENC_F_IMM_I: rv_inst->imm = (int32_t) inst >> 20; break;
    case ENC_F_IMM_S: rv_inst->imm = ((int32_t) inst >> 20 & ~0x1f) | UBFX(inst, 7, 11); break;
    case ENC_F_IMM_J:
        rv_inst->imm = (inst & 0xff000) | (inst >> (20-11) & (1 << 11)) |
                        (inst >> 11 & (1 << 20)) |
                        ((int32_t) inst >> (30 - 10) & 0xffe007fe);
        break;
    case ENC_F_IMM_B:
        rv_inst->imm = ((int32_t) inst >> (31-12) & 0xfffff000) |
                        (inst << (11-7) & (1 << 11)) | (inst >> (30-10) & 0x7e0) |
                        (inst >> (11 - 4) & 0x1e);
        break;
    case ENC_F_IMM_SHAMT: rv_inst->imm = UBFX(inst, 20, 25); break;
    case ENC_F_IMM_AMO: rv_inst->imm = UBFX(inst, 25, 26); break;
    }

    return 4;
}

static int rv_decode2(u16 inst, RvOptions opt, Insn(RV)* restrict rv_inst) {
    enum {
        ENC_I_NONE = 0,
        ENC_U_53_76,
        ENC_U_53_26,
        ENC_U_5_42_76,
        ENC_U_5_43_86,
        ENC_U_52_76,
        ENC_U_53_86,
        ENC_I_5_40,
        ENC_I_11_4_98_10_6_7_31_5,
        ENC_I_8_43_76_21_5,
        ENC_I_EBREAK,

        ENC_NZ_START,
        ENC_NZU_54_96_2_3,
        ENC_NZI_5_40,
        ENC_NZU_5_40,
        ENC_NZI_9_4_6_87_5,
        ENC_NZI_17_1612,
    } imm_enc = ENC_I_NONE;
    unsigned rs2f = UBFX(inst, 2, 6);
    unsigned rs2c = UBFX(inst, 2, 4) + 8;
    unsigned rs1f = UBFX(inst, 7, 11);
    unsigned rs1c = UBFX(inst, 7, 9) + 8;
    unsigned mnem = 0, rd = RV_REG_INV, rs1 = RV_REG_INV, rs2 = RV_REG_INV;
    switch (inst & 0xe003) {
    case 0x0000:
        mnem = RV_ADDI, rd = rs2c, rs1 = 2, imm_enc = ENC_NZU_54_96_2_3;
        break;
    case 0x2000:
        if ((opt & RV_RVMSK) == RV_RV128)
        return RV_UNDEF; // TODO
        mnem = RV_FLD, rd = rs2c, rs1 = rs1c, imm_enc = ENC_U_53_76;
        break;
    case 0x4000:
        mnem = RV_LW, rd = rs2c, rs1 = rs1c, imm_enc = ENC_U_53_26;
        break;
    case 0x6000:
        if ((opt & RV_RVMSK) == RV_RV32)
        mnem = RV_FLW, rd = rs2c, rs1 = rs1c, imm_enc = ENC_U_53_26;
        else
        mnem = RV_LD, rd = rs2c, rs1 = rs1c, imm_enc = ENC_U_53_76;
        break;
    case 0xa000:
        if ((opt & RV_RVMSK) != RV_RV128)
        mnem = RV_FSD, rs1 = rs1c, rs2 = rs2c, imm_enc = ENC_U_53_76;
        // TODO: RV128 C.SQ
        break;
    case 0xc000:
        mnem = RV_SW, rs1 = rs1c, rs2 = rs2c, imm_enc = ENC_U_53_26;
        break;
    case 0xe000:
        if ((opt & RV_RVMSK) == RV_RV32)
        mnem = RV_FSW, rs1 = rs1c, rs2 = rs2c, imm_enc = ENC_U_53_26;
        else
        mnem = RV_SD, rs1 = rs1c, rs2 = rs2c, imm_enc = ENC_U_53_76;
        break;

    case 0x0001:
        mnem = RV_ADDI, rd = rs1f, rs1 = rs1f, imm_enc = ENC_NZI_5_40;
        break;
    case 0x2001:
        if ((opt & RV_RVMSK) == RV_RV32) {
        mnem = RV_JAL, rd = 1, imm_enc = ENC_I_11_4_98_10_6_7_31_5;
        } else {
        if (rs1f == 0)
            return RV_UNDEF;
        mnem = RV_ADDIW, rd = rs1f, rs1 = rs1f, imm_enc = ENC_I_5_40;
        }
        break;
    case 0x4001:
        if (rs1f == 0)
        return RV_UNDEF;
        mnem = RV_ADDI, rd = rs1f, rs1 = 0, imm_enc = ENC_I_5_40;
        break;
    case 0x6001:
        if (rs1f == 0)
        return RV_UNDEF;
        if (rs1f == 2)
        mnem = RV_ADDI, rd = rs1f, rs1 = rs1f, imm_enc = ENC_NZI_9_4_6_87_5;
        else
        mnem = RV_LUI, rd = rs1f, imm_enc = ENC_NZI_17_1612;
        break;
    case 0x8001:
        switch (UBFX(inst, 10, 11)) {
        case 0: mnem = RV_SRLI, rd = rs1c, rs1 = rs1c, imm_enc = ENC_NZU_5_40; break;
        case 1: mnem = RV_SRAI, rd = rs1c, rs1 = rs1c, imm_enc = ENC_NZU_5_40; break;
        case 2: mnem = RV_ANDI, rd = rs1c, rs1 = rs1c, imm_enc = ENC_I_5_40; break;
        case 3:
        rd = rs1c, rs1 = rs1c, rs2 = rs2c, mnem = UBFX(inst, 5, 6) | (UBFX(inst, 12, 12) << 2);
        mnem = (const u16[8]) {RV_SUB, RV_XOR, RV_OR, RV_AND, RV_SUBW, RV_ADDW}[mnem];
        break;
        }
        break;
    case 0xa001:
        mnem = RV_JAL, rd = 0, imm_enc = ENC_I_11_4_98_10_6_7_31_5;
        break;
    case 0xc001:
        mnem = RV_BEQ, rs1 = rs1c, rs2 = 0, imm_enc = ENC_I_8_43_76_21_5;
        break;
    case 0xe001:
        mnem = RV_BNE, rs1 = rs1c, rs2 = 0, imm_enc = ENC_I_8_43_76_21_5;
        break;

    case 0x0002:
        mnem = RV_SLLI, rd = rs1f, rs1 = rs1f, imm_enc = ENC_NZU_5_40;
        break;
    case 0x2002:
        if ((opt & RV_RVMSK) != RV_RV128)
        mnem = RV_FLD, rd = rs1f, rs1 = 2, imm_enc = ENC_U_5_43_86;
        // TODO: RV128 C.LQSP
        break;
    case 0x4002:
        mnem = RV_LW, rd = rs1f, rs1 = 2, imm_enc = ENC_U_5_42_76;
        break;
    case 0x6002:
        if ((opt & RV_RVMSK) == RV_RV32)
        mnem = RV_FLW, rd = rs1f, rs1 = 2, imm_enc = ENC_U_5_42_76;
        else
        mnem = RV_LD, rd = rs1f, rs1 = 2, imm_enc = ENC_U_5_43_86;
        break;
    case 0x8002:
        if (!(inst & 0x1000)) {
        if (rs1f == 0)
            return RV_UNDEF;
        if (rs2f == 0)
            mnem = RV_JALR, rd = 0, rs1 = rs1f;
        else
            mnem = RV_ADD, rd = rs1f, rs1 = 0, rs2 = rs2f;
        } else {
        if (rs1f == 0)
            mnem = RV_ECALL, imm_enc = ENC_I_EBREAK;
        else if (rs2f == 0)
            mnem = RV_JALR, rd = 1, rs1 = rs1f;
        else
            mnem = RV_ADD, rd = rs1f, rs1 = rs1f, rs2 = rs2f;
        }
        break;
    case 0xa002:
        if ((opt & RV_RVMSK) != RV_RV128)
        mnem = RV_FSD, rs1 = 2, rs2 = rs2f, imm_enc = ENC_U_53_86;
        // TODO: RV128 C.LQSP
        break;
    case 0xc002:
        mnem = RV_SW, rs1 = 2, rs2 = rs2f, imm_enc = ENC_U_52_76;
        break;
    case 0xe002:
        if ((opt & RV_RVMSK) == RV_RV32)
        mnem = RV_FSW, rs1 = 2, rs2 = rs2f, imm_enc = ENC_U_52_76;
        else
        mnem = RV_SD, rs1 = 2, rs2 = rs2f, imm_enc = ENC_U_53_86;
        break;
    }

    if (!mnem)
        return RV_UNDEF;
    rv_inst->id = mnem;
    rv_inst->rd = rd;
    rv_inst->rs1 = rs1;
    rv_inst->rs2 = rs2;
    rv_inst->rs3 = RV_REG_INV;
    switch (imm_enc) {
    case ENC_I_NONE: rv_inst->imm = 0; break;
    case ENC_I_EBREAK: rv_inst->imm = 1; break;
    case ENC_I_5_40: rv_inst->imm = SBFXIZ(inst, 12, 12, 5) | UBFX(inst, 2, 6); break;
    case ENC_NZI_9_4_6_87_5:
        rv_inst->imm = SBFXIZ(inst, 12, 12, 9) | UBFX(inst, 3, 4) << 7 |
                        UBFX(inst, 5, 5) << 6 | UBFX(inst, 2, 2) << 5 |
                        UBFX(inst, 6, 6) << 4; break;
    case ENC_NZU_54_96_2_3:
        rv_inst->imm = UBFX(inst, 7, 10) << 6 | UBFX(inst, 11, 12) << 4 |
                        UBFX(inst, 5, 5) << 3 | UBFX(inst, 6, 6) << 2; break;
    case ENC_U_53_76:
        rv_inst->imm = UBFX(inst, 5, 6) << 6 | UBFX(inst, 10, 12) << 3; break;
    case ENC_U_53_26:
        rv_inst->imm = UBFX(inst, 5, 5) << 6 | UBFX(inst, 10, 12) << 3 |
                        UBFX(inst, 6, 6) << 2; break;
    case ENC_U_5_43_86:
        rv_inst->imm = UBFX(inst, 2, 4) << 6 | UBFX(inst, 12, 12) << 5 |
                        UBFX(inst, 5, 6) << 3; break;
    case ENC_U_5_42_76:
        rv_inst->imm = UBFX(inst, 2, 3) << 6 | UBFX(inst, 12, 12) << 5 |
                        UBFX(inst, 4, 6) << 2; break;
    case ENC_U_52_76:
        rv_inst->imm = UBFX(inst, 7, 8) << 6 | UBFX(inst, 9, 12) << 2; break;
    case ENC_U_53_86:
        rv_inst->imm = UBFX(inst, 7, 9) << 6 | UBFX(inst, 10, 12) << 3; break;
    case ENC_NZU_5_40:
        rv_inst->imm = UBFX(inst, 12, 12) << 5 | UBFX(inst, 2, 6); break;
    case ENC_NZI_5_40:
        rv_inst->imm = SBFXIZ(inst, 12, 12, 5) | UBFX(inst, 2, 6); break;
    case ENC_I_8_43_76_21_5:
        rv_inst->imm = SBFXIZ(inst, 12, 12, 8) | UBFX(inst, 5, 6) << 6 |
                        UBFX(inst, 2, 2) << 5 | UBFX(inst, 10, 11) << 3 |
                        UBFX(inst, 3, 4) << 1; break;
    case ENC_I_11_4_98_10_6_7_31_5:
        rv_inst->imm = SBFXIZ(inst, 12, 12, 11) | UBFX(inst, 8, 8) << 10 |
                        UBFX(inst, 9, 10) << 8 | UBFX(inst, 6, 6) << 7 |
                        UBFX(inst, 7, 7) << 6 | UBFX(inst, 2, 2) << 5 |
                        UBFX(inst, 11, 11) << 4 | UBFX(inst, 3, 5) << 1; break;
    case ENC_NZI_17_1612:
        rv_inst->imm = SBFXIZ(inst, 12, 12, 17) | UBFX(inst, 2, 6) << 12; break;
    default: return RV_UNDEF;
    }

    if (imm_enc >= ENC_NZ_START && !rv_inst->imm) {
        /* For most nzimm RV64C instructions, an immediate == 0 is invalid or reserved, but
        * for C.ADDI when rd == x0 it is used to encode a NOP instruction
        */
        if (rv_inst->rd != 0 || rv_inst->id != RV_ADDI) {
        return RV_UNDEF;
        }
    }
    return 2;
}

int rv_decode(size_t bufsz, const uint8_t* buf, RvOptions opt, Insn(RV)* restrict rv_inst) {
    if (UNLIKELY(bufsz < 2))
        return RV_PARTIAL;
    if ((buf[0] & 0x03) != 0x03)
        return rv_decode2(LOAD_LE_2(buf), opt, rv_inst);
    if ((buf[0] & 0x1c) != 0x1c) {
        if (UNLIKELY(bufsz < 4))
        return RV_PARTIAL;
        return rv_decode4(LOAD_LE_4(buf), rv_inst);
    }
    return RV_UNDEF; // instruction length > 32 bit
}

static void strlcat(char* restrict dst, const char* src, size_t size) {
    while (*dst && size)
        dst++, size--;
    while (*src && size > 1)
        *dst++ = *src++, size--;
    if (size)
        *dst = 0;
}

static char* rv_format_int(int32_t val, char buf[static 16]) {
    int32_t absneg = val < 0 ? val : -val; // avoid overflow
    unsigned idx = 16;
    buf[--idx] = 0;
    do {
        buf[--idx] = '0' - (absneg % 10);
        absneg /= 10;
    } while (absneg);
    if (val < 0)
        buf[--idx] = '-';
    return &buf[idx];
}

static const char* mnem_str[] = {
    [RV_LB] = "lb", [RV_LH] = "lh", [RV_LW] = "lw", [RV_LD] = "ld",
    [RV_LBU] = "lbu", [RV_LHU] = "lhu", [RV_LWU] = "lwu",
    [RV_SB] = "sb", [RV_SH] = "sh", [RV_SW] = "sw", [RV_SD] = "sd",

    [RV_ADDI] = "addi", [RV_SLLI] = "slli", [RV_SLTI] = "slti",
    [RV_SLTIU] = "sltiu", [RV_XORI] = "xori", [RV_SRAI] = "srai",
    [RV_SRLI] = "srli", [RV_ORI] = "ori", [RV_ANDI] = "andi",
    [RV_ADD] = "add", [RV_SLL] = "sll", [RV_SLT] = "slt",
    [RV_SLTU] = "sltu", [RV_XOR] = "xor", [RV_SRL] = "srl", [RV_OR] = "or",
    [RV_AND] = "and", [RV_SUB] = "sub", [RV_SRA] = "sra",

    [RV_FENCE] = "fence", [RV_FENCEI] = "fencei",
    [RV_AUIPC] = "auipc", [RV_LUI] = "lui",
    [RV_JAL] = "jal", [RV_JALR] = "jalr",
    [RV_BEQ] = "beq", [RV_BNE] = "bne", [RV_BLT] = "blt", [RV_BGE] = "bge",
    [RV_BLTU] = "bltu", [RV_BGEU] = "bgeu",
    [RV_ECALL] = "ecall",

    [RV_ADDIW] = "addiw", [RV_SLLIW] = "slliw", [RV_SRAIW] = "sraiw",
    [RV_SRLIW] = "srliw", [RV_ADDW] = "addw", [RV_SLLW] = "sllw",
    [RV_SRLW] = "srlw", [RV_SUBW] = "subw", [RV_SRAW] = "sraw",

    [RV_MUL] = "mul", [RV_MULH] = "mulh", [RV_MULHSU] = "mulhsu",
    [RV_MULHU] = "mulhu", [RV_DIV] = "div", [RV_DIVU] = "divu",
    [RV_REM] = "rem", [RV_REMU] = "remu", [RV_MULW] = "mulw",
    [RV_DIVW] = "divw", [RV_DIVUW] = "divuw", [RV_REMW] = "remw",
    [RV_REMUW] = "remuw",

    [RV_LRW] = "lr.w", [RV_SCW] = "sc.w",
    [RV_LRD] = "lr.d", [RV_SCD] = "sc.d",
    [RV_AMOADDW] = "amoadd.w", [RV_AMOADDD] = "amoadd.d",
    [RV_AMOSWAPW] = "amoswap.w", [RV_AMOSWAPD] = "amoswap.d",
    [RV_AMOXORW] = "amoxor.w", [RV_AMOXORD] = "amoxor.d",
    [RV_AMOORW] = "amoor.w", [RV_AMOORD] = "amoor.d",
    [RV_AMOANDW] = "amoand.w", [RV_AMOANDD] = "amoand.d",
    [RV_AMOMINW] = "amomin.w", [RV_AMOMIND] = "amomin.d",
    [RV_AMOMAXW] = "amomax.w", [RV_AMOMAXD] = "amomax.d",
    [RV_AMOMINUW] = "amominu.w", [RV_AMOMINUD] = "amominu.d",
    [RV_AMOMAXUW] = "amomaxu.w", [RV_AMOMAXUD] = "amomaxu.d",

    [RV_CSRRW] = "csrrw", [RV_CSRRWI] = "csrrwi",
    [RV_CSRRS] = "csrrs", [RV_CSRRSI] = "csrrsi",
    [RV_CSRRC] = "csrrc", [RV_CSRRCI] = "csrrci",

    [RV_FLW] = "flw", [RV_FSW] = "fsw",
    [RV_FMVXW] = "fmv.x.w", [RV_FMVWX] = "fmv.w.x", [RV_FCLASSS] = "fclass.s",
    [RV_FMADDS] = "fmadd.s", [RV_FMSUBS] = "fmsub.s",
    [RV_FNMSUBS] = "fnmsub.s", [RV_FNMADDS] = "fnmadd.s",
    [RV_FADDS] = "fadd.s", [RV_FSUBS] = "fsub.s",
    [RV_FMULS] = "fmul.s", [RV_FDIVS] = "fdiv.s", [RV_FSQRTS] = "fsqrt.s",
    [RV_FSGNJS] = "fsgnj.s", [RV_FSGNJNS] = "fsgnjn.s",
    [RV_FSGNJXS] = "fsgnjx.s", [RV_FMINS] = "fmin.s", [RV_FMAXS] = "fmax.s",
    [RV_FLES] = "fle.s", [RV_FLTS] = "flt.s", [RV_FEQS] = "feq.s",
    [RV_FCVTWS] = "fcvt.w.s", [RV_FCVTWUS] = "fcvt.wu.s",
    [RV_FCVTLS] = "fcvt.l.s", [RV_FCVTLUS] = "fcvt.lu.s",
    [RV_FCVTSW] = "fcvt.s.w", [RV_FCVTSWU] = "fcvt.s.wu",
    [RV_FCVTSL] = "fcvt.s.l", [RV_FCVTSLU] = "fcvt.s.lu",
    // RV32D/RV64D
    [RV_FLD] = "fld", [RV_FSD] = "fsd",
    [RV_FMVXD] = "fmv.x.d", [RV_FMVDX] = "fmv.d.x", [RV_FCLASSD] = "fclass.d",
    [RV_FMADDD] = "fmadd.d", [RV_FMSUBD] = "fmsub.d",
    [RV_FNMSUBD] = "fnmsub.d", [RV_FNMADDD] = "fnmadd.d",
    [RV_FADDD] = "fadd.d", [RV_FSUBD] = "fsub.d",
    [RV_FMULD] = "fmul.d", [RV_FDIVD] = "fdiv.d", [RV_FSQRTD] = "fsqrt.d",
    [RV_FSGNJD] = "fsgnj.d", [RV_FSGNJND] = "fsgnjn.d",
    [RV_FSGNJXD] = "fsgnjx.d", [RV_FMIND] = "fmin.d", [RV_FMAXD] = "fmax.d",
    [RV_FLED] = "fle.d", [RV_FLTD] = "flt.d", [RV_FEQD] = "feq.d",
    [RV_FCVTSD] = "fcvt.s.d", [RV_FCVTDS] = "fcvt.d.s",
    [RV_FCVTWD] = "fcvt.w.d", [RV_FCVTWUD] = "fcvt.wu.d",
    [RV_FCVTLD] = "fcvt.l.d", [RV_FCVTLUD] = "fcvt.lu.d",
    [RV_FCVTDW] = "fcvt.d.w", [RV_FCVTDWU] = "fcvt.d.wu",
    [RV_FCVTDL] = "fcvt.d.l", [RV_FCVTDLU] = "fcvt.d.lu",
};

void rv_format(const Insn(RV)* inst, size_t len, char* restrict buf) {
    char tmp[18];
    if (!len)
        return;
    buf[0] = 0;
    
    if (inst->id >= sizeof mnem_str / sizeof mnem_str[0] || !mnem_str[inst->id]) {
        strlcat(buf, "<invalid>", len);
        return;
    }
    strlcat(buf, mnem_str[inst->id], len);
    if (inst->rd != RV_REG_INV) {
        char* fmt = rv_format_int(inst->rd, tmp + 2);
        *--fmt = 'r';
        *--fmt = ' ';
        strlcat(buf, fmt, len);
    }
    if (inst->rs1 != RV_REG_INV) {
        char* fmt = rv_format_int(inst->rs1, tmp + 2);
        *--fmt = 'r';
        *--fmt = ' ';
        strlcat(buf, fmt, len);
    }
    if (inst->rs2 != RV_REG_INV) {
        char* fmt = rv_format_int(inst->rs2, tmp + 2);
        *--fmt = 'r';
        *--fmt = ' ';
        strlcat(buf, fmt, len);
    }
    if (inst->rs3 != RV_REG_INV) {
        char* fmt = rv_format_int(inst->rs3, tmp + 2);
        *--fmt = 'r';
        *--fmt = ' ';
        strlcat(buf, fmt, len);
    }
    if (inst->imm) {
        char* fmt = rv_format_int(inst->imm, tmp + 2);
        *--fmt = ' ';
        strlcat(buf, fmt, len);
    }
}



// ==================================================================================== //
//                                    eir: Block                                      
// ==================================================================================== //

Block_fn_def(RV);

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

#define RV_EXEC_I_M(S, A, L)                                  \
    do {                                                      \
        u8 rd = Val_as_u8(insn->oprs[0], 0);                  \
        u8 r1 = Val_as_u8(insn->oprs[1], 0);                  \
        i64 immi = Val_as_i64(insn->oprs[2], 0);              \
        Val* r1_v = CPUState_get_reg(RV, cpu, r1);            \
        Val* res = CPUState_get_mem(RV, cpu, A, L);           \
        CPUState_set_reg(RV, cpu, rd, res);                   \
        Val_free(r1_v);                                       \
        Val_free(res);                                        \
    } while (0)

#define RV_EXEC_U(...)                           \
    do {                                         \
        u8 rd = Val_as_u8(insn->oprs[0], 0);     \
        i64 immu = Val_as_i64(insn->oprs[1], 0); \
        Val* res = Val_new_i64(__VA_ARGS__);     \
        CPUState_set_reg(RV, cpu, rd, res);      \
        Val_free(res);                           \
    } while (0)

#define RV_EXEC_S_M(A, N)                          \
    do {                                           \
        u8 r1 = Val_as_u8(insn->oprs[0], 0);       \
        u8 r2 = Val_as_u8(insn->oprs[1], 0);       \
        i64 imms = Val_as_i64(insn->oprs[2], 0);   \
        Val* r1_v = CPUState_get_reg(RV, cpu, r1); \
        Val* r2_v = CPUState_get_reg(RV, cpu, r2); \
        CPUState_set_mem(RV, cpu, A, r2_v, N);     \
        Val_free(r1_v);                            \
        Val_free(r2_v);                            \
    } while (0)

void CPUState_OP_def(RV, execute)(CPUState(RV) * cpu, Insn(RV) * insn) {
    switch((int)insn->id) {
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
        case RV_LB      : RV_EXEC_I_M(i, Val_new_i64(Val_as_i64(r1_v, 0) + immi), 1);   break;
        case RV_LH      : RV_EXEC_I_M(i, Val_new_i64(Val_as_i64(r1_v, 0) + immi), 2);   break;
        case RV_LW      : RV_EXEC_I_M(i, Val_new_i64(Val_as_i64(r1_v, 0) + immi), 4);   break;
        case RV_LBU     : RV_EXEC_I_M(u, Val_new_i64(Val_as_i64(r1_v, 0) + immi), 1);   break;
        case RV_LHU     : RV_EXEC_I_M(u, Val_new_i64(Val_as_i64(r1_v, 0) + immi), 2);   break;
        /* RV32I: Store S-Type */
        case RV_SB      : RV_EXEC_S_M(Val_new_i64(Val_as_i64(r1_v, 0) + imms), 1);      break;
        case RV_SH      : RV_EXEC_S_M(Val_new_i64(Val_as_i64(r1_v, 0) + imms), 2);      break;
        case RV_SW      : RV_EXEC_S_M(Val_new_i64(Val_as_i64(r1_v, 0) + imms), 4);      break;
        /* RV32I: Branch */

        /* RV32I: Device & System */
        case RV_EBREAK  : CPUState_set_status(RV, cpu, CPU_END, cpu->pc, Val_as_u64(CPUState_get_reg(RV, cpu, 10), 0)); break;
        default: break;
    }
    Val_copy(cpu->pc, cpu->dnpc);
}




