
#include "rvdec.h"

#include <stddef.h>
#include <stdint.h>

#define LOAD_LE_1(buf) ((size_t) *(uint8_t*) (buf))
#define LOAD_LE_2(buf) (LOAD_LE_1(buf) | LOAD_LE_1((uint8_t*) (buf) + 1)<<8)
#define LOAD_LE_4(buf) (LOAD_LE_2(buf) | LOAD_LE_2((uint8_t*) (buf) + 2)<<16)
#define UBFX(val, start, end) (((val) >> start) & ((1 << (end - start + 1)) - 1))
#define SBFXIZ(val, start, end, shl) ((struct { long v: end-start+1+shl; }) {UBFX(val, start, end)<<shl}.v)
#define UNLIKELY(x) __builtin_expect((x), 0)

static int frv_decode4(uint32_t inst, FrvInst* restrict frv_inst) {
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
    mnem = (const uint16_t[]) {FRV_LB, FRV_LH, FRV_LW, FRV_LD, FRV_LBU, FRV_LHU, FRV_LWU, 0}[funct3];
    break;
  case 0x01: encoding = ENC_I;
    mnem = (const uint16_t[]) {0, 0, FRV_FLW, FRV_FLD, 0, 0, 0, 0}[funct3];
    break;
  case 0x03: encoding = ENC_I;
    mnem = (const uint16_t[]) {FRV_FENCE, FRV_FENCEI, 0, 0, 0, 0, 0, 0}[funct3];
    break;
  case 0x04:
    switch (funct3) {
    case 0: encoding = ENC_I; mnem = FRV_ADDI; break;
    case 1: encoding = ENC_I_SHAMT; mnem = FRV_SLLI; break;
    case 2: encoding = ENC_I; mnem = FRV_SLTI; break;
    case 3: encoding = ENC_I; mnem = FRV_SLTIU; break;
    case 4: encoding = ENC_I; mnem = FRV_XORI; break;
    case 5: encoding = ENC_I_SHAMT; mnem = funct7 & 0x20 ? FRV_SRAI : FRV_SRLI; break;
    case 6: encoding = ENC_I; mnem = FRV_ORI; break;
    case 7: encoding = ENC_I; mnem = FRV_ANDI; break;
    }
    break;
  case 0x05: encoding = ENC_U; mnem = FRV_AUIPC; break;
  case 0x06:
    switch (funct3) {
    case 0: encoding = ENC_I; mnem = FRV_ADDIW; break;
    case 1: encoding = ENC_I_SHAMT; mnem = FRV_SLLIW; break;
    case 5: encoding = ENC_I_SHAMT; mnem = funct7 & 0x20 ? FRV_SRAIW : FRV_SRLIW; break;
    default: return FRV_UNDEF;
    }
    break;
  case 0x08: encoding = ENC_S;
    mnem = (const uint16_t[]) {FRV_SB, FRV_SH, FRV_SW, FRV_SD, 0, 0, 0, 0}[funct3];
    break;
  case 0x09: encoding = ENC_S;
    mnem = (const uint16_t[]) {0, 0, FRV_FSW, FRV_FSD, 0, 0, 0, 0}[funct3];
    break;
  case 0x0b: encoding = ENC_R | ENC_F_IMM_AMO;
    switch (funct7 >> 2) {
    case 0x00: mnem = (const uint16_t[]) {0, 0, FRV_AMOADDW, FRV_AMOADDD, 0, 0, 0, 0}[funct3]; break;
    case 0x01: mnem = (const uint16_t[]) {0, 0, FRV_AMOSWAPW, FRV_AMOSWAPD, 0, 0, 0, 0}[funct3]; break;
    case 0x02: mnem = (const uint16_t[]) {0, 0, FRV_LRW, FRV_LRD, 0, 0, 0, 0}[funct3]; break;
    case 0x03: mnem = (const uint16_t[]) {0, 0, FRV_SCW, FRV_SCD, 0, 0, 0, 0}[funct3]; break;
    case 0x04: mnem = (const uint16_t[]) {0, 0, FRV_AMOXORW, FRV_AMOXORD, 0, 0, 0, 0}[funct3]; break;
    case 0x08: mnem = (const uint16_t[]) {0, 0, FRV_AMOORW, FRV_AMOORD, 0, 0, 0, 0}[funct3]; break;
    case 0x0c: mnem = (const uint16_t[]) {0, 0, FRV_AMOANDW, FRV_AMOANDD, 0, 0, 0, 0}[funct3]; break;
    case 0x10: mnem = (const uint16_t[]) {0, 0, FRV_AMOMINW, FRV_AMOMIND, 0, 0, 0, 0}[funct3]; break;
    case 0x14: mnem = (const uint16_t[]) {0, 0, FRV_AMOMAXW, FRV_AMOMAXD, 0, 0, 0, 0}[funct3]; break;
    case 0x18: mnem = (const uint16_t[]) {0, 0, FRV_AMOMINUW, FRV_AMOMINUD, 0, 0, 0, 0}[funct3]; break;
    case 0x1c: mnem = (const uint16_t[]) {0, 0, FRV_AMOMAXUW, FRV_AMOMAXUD, 0, 0, 0, 0}[funct3]; break;
    default: return FRV_UNDEF;
    }
    break;
  case 0x0c: encoding = ENC_R;
    switch (funct7) {
    case 0x00: mnem = (const uint16_t[]) {FRV_ADD, FRV_SLL, FRV_SLT, FRV_SLTU, FRV_XOR, FRV_SRL, FRV_OR, FRV_AND}[funct3]; break;
    case 0x01: mnem = (const uint16_t[]) {FRV_MUL, FRV_MULH, FRV_MULHSU, FRV_MULHU, FRV_DIV, FRV_DIVU, FRV_REM, FRV_REMU}[funct3]; break;
    case 0x20: mnem = (const uint16_t[]) {FRV_SUB, 0, 0, 0, 0, FRV_SRA, 0, 0}[funct3]; break;
    default: return FRV_UNDEF;
    }
    break;
  case 0x0d: encoding = ENC_U; mnem = FRV_LUI; break;
  case 0x0e: encoding = ENC_R;
    switch (funct7) {
    case 0x00: mnem = (const uint16_t[]) {FRV_ADDW, FRV_SLLW, 0, 0, 0, FRV_SRLW, 0, 0}[funct3]; break;
    case 0x01: mnem = (const uint16_t[]) {FRV_MULW, 0, 0, 0, FRV_DIVW, FRV_DIVUW, FRV_REMW, FRV_REMUW}[funct3]; break;
    case 0x20: mnem = (const uint16_t[]) {FRV_SUBW, 0, 0, 0, 0, FRV_SRAW, 0, 0}[funct3]; break;
    default: return FRV_UNDEF;
    }
    break;
  case 0x10: encoding = ENC_R4 | ENC_F_RM; mnem = (const uint16_t[4]) {FRV_FMADDS, FRV_FMADDD}[UBFX(inst, 25, 26)]; break;
  case 0x11: encoding = ENC_R4 | ENC_F_RM; mnem = (const uint16_t[4]) {FRV_FMSUBS, FRV_FMSUBD}[UBFX(inst, 25, 26)]; break;
  case 0x12: encoding = ENC_R4 | ENC_F_RM; mnem = (const uint16_t[4]) {FRV_FNMSUBS, FRV_FNMSUBD}[UBFX(inst, 25, 26)]; break;
  case 0x13: encoding = ENC_R4 | ENC_F_RM; mnem = (const uint16_t[4]) {FRV_FNMADDS, FRV_FNMADDD}[UBFX(inst, 25, 26)]; break;
  case 0x14:
    switch (funct7) {
    case 0x00: encoding = ENC_R | ENC_F_RM; mnem = FRV_FADDS; break;
    case 0x01: encoding = ENC_R | ENC_F_RM; mnem = FRV_FADDD; break;
    case 0x04: encoding = ENC_R | ENC_F_RM; mnem = FRV_FSUBS; break;
    case 0x05: encoding = ENC_R | ENC_F_RM; mnem = FRV_FSUBD; break;
    case 0x08: encoding = ENC_R | ENC_F_RM; mnem = FRV_FMULS; break;
    case 0x09: encoding = ENC_R | ENC_F_RM; mnem = FRV_FMULD; break;
    case 0x0c: encoding = ENC_R | ENC_F_RM; mnem = FRV_FDIVS; break;
    case 0x0d: encoding = ENC_R | ENC_F_RM; mnem = FRV_FDIVD; break;
    case 0x20: encoding = ENC_R2 | ENC_F_RM; mnem = (const uint16_t[32]) {0, FRV_FCVTSD}[UBFX(inst, 20, 24)]; break;
    case 0x21: encoding = ENC_R2 | ENC_F_RM; mnem = (const uint16_t[32]) {FRV_FCVTDS}[UBFX(inst, 20, 24)]; break;
    case 0x2c: encoding = ENC_R2 | ENC_F_RM; mnem = FRV_FSQRTS; break; // TODO: check rs2
    case 0x2d: encoding = ENC_R2 | ENC_F_RM; mnem = FRV_FSQRTD; break; // TODO: check rs2
    case 0x10: encoding = ENC_R; mnem = (const uint16_t[]) {FRV_FSGNJS, FRV_FSGNJNS, FRV_FSGNJXS, 0, 0, 0, 0, 0}[funct3]; break;
    case 0x11: encoding = ENC_R; mnem = (const uint16_t[]) {FRV_FSGNJD, FRV_FSGNJND, FRV_FSGNJXD, 0, 0, 0, 0, 0}[funct3]; break;
    case 0x14: encoding = ENC_R; mnem = (const uint16_t[]) {FRV_FMINS, FRV_FMAXS, 0, 0, 0, 0, 0, 0}[funct3]; break;
    case 0x15: encoding = ENC_R; mnem = (const uint16_t[]) {FRV_FMIND, FRV_FMAXD, 0, 0, 0, 0, 0, 0}[funct3]; break;
    case 0x50: encoding = ENC_R; mnem = (const uint16_t[]) {FRV_FLES, FRV_FLTS, FRV_FEQS, 0, 0, 0, 0, 0}[funct3]; break;
    case 0x51: encoding = ENC_R; mnem = (const uint16_t[]) {FRV_FLED, FRV_FLTD, FRV_FEQD, 0, 0, 0, 0, 0}[funct3]; break;
    case 0x60: encoding = ENC_R2 | ENC_F_RM; mnem = (const uint16_t[32]) {FRV_FCVTWS, FRV_FCVTWUS, FRV_FCVTLS, FRV_FCVTLUS}[UBFX(inst, 20, 24)]; break;
    case 0x61: encoding = ENC_R2 | ENC_F_RM; mnem = (const uint16_t[32]) {FRV_FCVTWD, FRV_FCVTWUD, FRV_FCVTLD, FRV_FCVTLUD}[UBFX(inst, 20, 24)]; break;
    case 0x68: encoding = ENC_R2 | ENC_F_RM; mnem = (const uint16_t[32]) {FRV_FCVTSW, FRV_FCVTSWU, FRV_FCVTSL, FRV_FCVTSLU}[UBFX(inst, 20, 24)]; break;
    case 0x69: encoding = ENC_R2 | ENC_F_RM; mnem = (const uint16_t[32]) {FRV_FCVTDW, FRV_FCVTDWU, FRV_FCVTDL, FRV_FCVTDLU}[UBFX(inst, 20, 24)]; break;
    case 0x70: encoding = ENC_R2; mnem = (const uint16_t[]) {FRV_FMVXW, FRV_FCLASSS, 0, 0, 0, 0, 0, 0}[funct3]; break; // TODO: check rs2
    case 0x71: encoding = ENC_R2; mnem = (const uint16_t[]) {FRV_FMVXD, FRV_FCLASSD, 0, 0, 0, 0, 0, 0}[funct3]; break; // TODO: check rs2
    case 0x78: encoding = ENC_R2; mnem = (const uint16_t[]) {FRV_FMVWX, 0, 0, 0, 0, 0, 0, 0}[funct3]; break; // TODO: check rs2
    case 0x79: encoding = ENC_R2; mnem = (const uint16_t[]) {FRV_FMVDX, 0, 0, 0, 0, 0, 0, 0}[funct3]; break; // TODO: check rs2
    default: return FRV_UNDEF;
    }
    break;
  case 0x18: encoding = ENC_B;
    mnem = (const uint16_t[]) {FRV_BEQ, FRV_BNE, 0, 0, FRV_BLT, FRV_BGE, FRV_BLTU, FRV_BGEU}[funct3];
    break;
  case 0x19: encoding = ENC_I; mnem = FRV_JALR; break; // TODO: check funct3
  case 0x1b: encoding = ENC_J; mnem = FRV_JAL; break;
  case 0x1c: encoding = ENC_I;
    mnem = (const uint16_t[]) {FRV_ECALL, FRV_CSRRW, FRV_CSRRS, FRV_CSRRC, 0, FRV_CSRRWI, FRV_CSRRSI, FRV_CSRRCI}[funct3];
    break;
  }

  if (!mnem)
    return FRV_UNDEF;
  frv_inst->mnem = mnem;
  frv_inst->rd = (encoding & ENC_F_RD) ? UBFX(inst, 7, 11) : FRV_REG_INV;
  frv_inst->rs1 = (encoding & ENC_F_RS1) ? UBFX(inst, 15, 19) : FRV_REG_INV;
  frv_inst->rs2 = (encoding & ENC_F_RS2) ? UBFX(inst, 20, 24) : FRV_REG_INV;
  frv_inst->rs3 = (encoding & ENC_F_RS3) ? UBFX(inst, 27, 31) : FRV_REG_INV;
  if (encoding & ENC_F_RM)
    frv_inst->misc = funct3;
  switch (encoding & ENC_F_IMM_MASK) {
  default: frv_inst->imm = 0; break;
  case ENC_F_IMM_U: frv_inst->imm = UBFX(inst, 12, 31) << 12; break;
  case ENC_F_IMM_I: frv_inst->imm = (int32_t) inst >> 20; break;
  case ENC_F_IMM_S: frv_inst->imm = ((int32_t) inst >> 20 & ~0x1f) | UBFX(inst, 7, 11); break;
  case ENC_F_IMM_J:
    frv_inst->imm = (inst & 0xff000) | (inst >> (20-11) & (1 << 11)) |
                    (inst >> 11 & (1 << 20)) |
                    ((int32_t) inst >> (30 - 10) & 0xffe007fe);
    break;
  case ENC_F_IMM_B:
    frv_inst->imm = ((int32_t) inst >> (31-12) & 0xfffff000) |
                    (inst << (11-7) & (1 << 11)) | (inst >> (30-10) & 0x7e0) |
                    (inst >> (11 - 4) & 0x1e);
    break;
  case ENC_F_IMM_SHAMT: frv_inst->imm = UBFX(inst, 20, 25); break;
  case ENC_F_IMM_AMO: frv_inst->imm = UBFX(inst, 25, 26); break;
  }

  return 4;
}

static int frv_decode2(uint16_t inst, FrvOptions opt,
                       FrvInst* restrict frv_inst) {
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
  unsigned mnem = 0, rd = FRV_REG_INV, rs1 = FRV_REG_INV, rs2 = FRV_REG_INV;
  switch (inst & 0xe003) {
  case 0x0000:
    mnem = FRV_ADDI, rd = rs2c, rs1 = 2, imm_enc = ENC_NZU_54_96_2_3;
    break;
  case 0x2000:
    if ((opt & FRV_RVMSK) == FRV_RV128)
      return FRV_UNDEF; // TODO
    mnem = FRV_FLD, rd = rs2c, rs1 = rs1c, imm_enc = ENC_U_53_76;
    break;
  case 0x4000:
    mnem = FRV_LW, rd = rs2c, rs1 = rs1c, imm_enc = ENC_U_53_26;
    break;
  case 0x6000:
    if ((opt & FRV_RVMSK) == FRV_RV32)
      mnem = FRV_FLW, rd = rs2c, rs1 = rs1c, imm_enc = ENC_U_53_26;
    else
      mnem = FRV_LD, rd = rs2c, rs1 = rs1c, imm_enc = ENC_U_53_76;
    break;
  case 0xa000:
    if ((opt & FRV_RVMSK) != FRV_RV128)
      mnem = FRV_FSD, rs1 = rs1c, rs2 = rs2c, imm_enc = ENC_U_53_76;
    // TODO: RV128 C.SQ
    break;
  case 0xc000:
    mnem = FRV_SW, rs1 = rs1c, rs2 = rs2c, imm_enc = ENC_U_53_26;
    break;
  case 0xe000:
    if ((opt & FRV_RVMSK) == FRV_RV32)
      mnem = FRV_FSW, rs1 = rs1c, rs2 = rs2c, imm_enc = ENC_U_53_26;
    else
      mnem = FRV_SD, rs1 = rs1c, rs2 = rs2c, imm_enc = ENC_U_53_76;
    break;

  case 0x0001:
    mnem = FRV_ADDI, rd = rs1f, rs1 = rs1f, imm_enc = ENC_NZI_5_40;
    break;
  case 0x2001:
    if ((opt & FRV_RVMSK) == FRV_RV32) {
      mnem = FRV_JAL, rd = 1, imm_enc = ENC_I_11_4_98_10_6_7_31_5;
    } else {
      if (rs1f == 0)
        return FRV_UNDEF;
      mnem = FRV_ADDIW, rd = rs1f, rs1 = rs1f, imm_enc = ENC_I_5_40;
    }
    break;
  case 0x4001:
    if (rs1f == 0)
      return FRV_UNDEF;
    mnem = FRV_ADDI, rd = rs1f, rs1 = 0, imm_enc = ENC_I_5_40;
    break;
  case 0x6001:
    if (rs1f == 0)
      return FRV_UNDEF;
    if (rs1f == 2)
      mnem = FRV_ADDI, rd = rs1f, rs1 = rs1f, imm_enc = ENC_NZI_9_4_6_87_5;
    else
      mnem = FRV_LUI, rd = rs1f, imm_enc = ENC_NZI_17_1612;
    break;
  case 0x8001:
    switch (UBFX(inst, 10, 11)) {
    case 0: mnem = FRV_SRLI, rd = rs1c, rs1 = rs1c, imm_enc = ENC_NZU_5_40; break;
    case 1: mnem = FRV_SRAI, rd = rs1c, rs1 = rs1c, imm_enc = ENC_NZU_5_40; break;
    case 2: mnem = FRV_ANDI, rd = rs1c, rs1 = rs1c, imm_enc = ENC_I_5_40; break;
    case 3:
      rd = rs1c, rs1 = rs1c, rs2 = rs2c, mnem = UBFX(inst, 5, 6) | (UBFX(inst, 12, 12) << 2);
      mnem = (const uint16_t[8]) {FRV_SUB, FRV_XOR, FRV_OR, FRV_AND, FRV_SUBW, FRV_ADDW}[mnem];
      break;
    }
    break;
  case 0xa001:
    mnem = FRV_JAL, rd = 0, imm_enc = ENC_I_11_4_98_10_6_7_31_5;
    break;
  case 0xc001:
    mnem = FRV_BEQ, rs1 = rs1c, rs2 = 0, imm_enc = ENC_I_8_43_76_21_5;
    break;
  case 0xe001:
    mnem = FRV_BNE, rs1 = rs1c, rs2 = 0, imm_enc = ENC_I_8_43_76_21_5;
    break;

  case 0x0002:
    mnem = FRV_SLLI, rd = rs1f, rs1 = rs1f, imm_enc = ENC_NZU_5_40;
    break;
  case 0x2002:
    if ((opt & FRV_RVMSK) != FRV_RV128)
      mnem = FRV_FLD, rd = rs1f, rs1 = 2, imm_enc = ENC_U_5_43_86;
    // TODO: RV128 C.LQSP
    break;
  case 0x4002:
    mnem = FRV_LW, rd = rs1f, rs1 = 2, imm_enc = ENC_U_5_42_76;
    break;
  case 0x6002:
    if ((opt & FRV_RVMSK) == FRV_RV32)
      mnem = FRV_FLW, rd = rs1f, rs1 = 2, imm_enc = ENC_U_5_42_76;
    else
      mnem = FRV_LD, rd = rs1f, rs1 = 2, imm_enc = ENC_U_5_43_86;
    break;
  case 0x8002:
    if (!(inst & 0x1000)) {
      if (rs1f == 0)
        return FRV_UNDEF;
      if (rs2f == 0)
        mnem = FRV_JALR, rd = 0, rs1 = rs1f;
      else
        mnem = FRV_ADD, rd = rs1f, rs1 = 0, rs2 = rs2f;
    } else {
      if (rs1f == 0)
        mnem = FRV_ECALL, imm_enc = ENC_I_EBREAK;
      else if (rs2f == 0)
        mnem = FRV_JALR, rd = 1, rs1 = rs1f;
      else
        mnem = FRV_ADD, rd = rs1f, rs1 = rs1f, rs2 = rs2f;
    }
    break;
  case 0xa002:
    if ((opt & FRV_RVMSK) != FRV_RV128)
      mnem = FRV_FSD, rs1 = 2, rs2 = rs2f, imm_enc = ENC_U_53_86;
    // TODO: RV128 C.LQSP
    break;
  case 0xc002:
    mnem = FRV_SW, rs1 = 2, rs2 = rs2f, imm_enc = ENC_U_52_76;
    break;
  case 0xe002:
    if ((opt & FRV_RVMSK) == FRV_RV32)
      mnem = FRV_FSW, rs1 = 2, rs2 = rs2f, imm_enc = ENC_U_52_76;
    else
      mnem = FRV_SD, rs1 = 2, rs2 = rs2f, imm_enc = ENC_U_53_86;
    break;
  }

  if (!mnem)
    return FRV_UNDEF;
  frv_inst->mnem = mnem;
  frv_inst->rd = rd;
  frv_inst->rs1 = rs1;
  frv_inst->rs2 = rs2;
  frv_inst->rs3 = FRV_REG_INV;
  switch (imm_enc) {
  case ENC_I_NONE: frv_inst->imm = 0; break;
  case ENC_I_EBREAK: frv_inst->imm = 1; break;
  case ENC_I_5_40: frv_inst->imm = SBFXIZ(inst, 12, 12, 5) | UBFX(inst, 2, 6); break;
  case ENC_NZI_9_4_6_87_5:
    frv_inst->imm = SBFXIZ(inst, 12, 12, 9) | UBFX(inst, 3, 4) << 7 |
                    UBFX(inst, 5, 5) << 6 | UBFX(inst, 2, 2) << 5 |
                    UBFX(inst, 6, 6) << 4; break;
  case ENC_NZU_54_96_2_3:
    frv_inst->imm = UBFX(inst, 7, 10) << 6 | UBFX(inst, 11, 12) << 4 |
                    UBFX(inst, 5, 5) << 3 | UBFX(inst, 6, 6) << 2; break;
  case ENC_U_53_76:
    frv_inst->imm = UBFX(inst, 5, 6) << 6 | UBFX(inst, 10, 12) << 3; break;
  case ENC_U_53_26:
    frv_inst->imm = UBFX(inst, 5, 5) << 6 | UBFX(inst, 10, 12) << 3 |
                    UBFX(inst, 6, 6) << 2; break;
  case ENC_U_5_43_86:
    frv_inst->imm = UBFX(inst, 2, 4) << 6 | UBFX(inst, 12, 12) << 5 |
                    UBFX(inst, 5, 6) << 3; break;
  case ENC_U_5_42_76:
    frv_inst->imm = UBFX(inst, 2, 3) << 6 | UBFX(inst, 12, 12) << 5 |
                    UBFX(inst, 4, 6) << 2; break;
  case ENC_U_52_76:
    frv_inst->imm = UBFX(inst, 7, 8) << 6 | UBFX(inst, 9, 12) << 2; break;
  case ENC_U_53_86:
    frv_inst->imm = UBFX(inst, 7, 9) << 6 | UBFX(inst, 10, 12) << 3; break;
  case ENC_NZU_5_40:
    frv_inst->imm = UBFX(inst, 12, 12) << 5 | UBFX(inst, 2, 6); break;
  case ENC_NZI_5_40:
    frv_inst->imm = SBFXIZ(inst, 12, 12, 5) | UBFX(inst, 2, 6); break;
  case ENC_I_8_43_76_21_5:
    frv_inst->imm = SBFXIZ(inst, 12, 12, 8) | UBFX(inst, 5, 6) << 6 |
                    UBFX(inst, 2, 2) << 5 | UBFX(inst, 10, 11) << 3 |
                    UBFX(inst, 3, 4) << 1; break;
  case ENC_I_11_4_98_10_6_7_31_5:
    frv_inst->imm = SBFXIZ(inst, 12, 12, 11) | UBFX(inst, 8, 8) << 10 |
                    UBFX(inst, 9, 10) << 8 | UBFX(inst, 6, 6) << 7 |
                    UBFX(inst, 7, 7) << 6 | UBFX(inst, 2, 2) << 5 |
                    UBFX(inst, 11, 11) << 4 | UBFX(inst, 3, 5) << 1; break;
  case ENC_NZI_17_1612:
    frv_inst->imm = SBFXIZ(inst, 12, 12, 17) | UBFX(inst, 2, 6) << 12; break;
  default: return FRV_UNDEF;
  }

  if (imm_enc >= ENC_NZ_START && !frv_inst->imm) {
    /* For most nzimm RV64C instructions, an immediate == 0 is invalid or reserved, but
     * for C.ADDI when rd == x0 it is used to encode a NOP instruction
     */

    if (frv_inst->rd != 0 || frv_inst->mnem != FRV_ADDI) {
      return FRV_UNDEF;
    }
  }
  return 2;
}

int frv_decode(size_t bufsz, const uint8_t* buf, FrvOptions opt,
               FrvInst* restrict frv_inst) {
  if (UNLIKELY(bufsz < 2))
    return FRV_PARTIAL;
  if ((buf[0] & 0x03) != 0x03)
    return frv_decode2(LOAD_LE_2(buf), opt, frv_inst);
  if ((buf[0] & 0x1c) != 0x1c) {
    if (UNLIKELY(bufsz < 4))
      return FRV_PARTIAL;
    return frv_decode4(LOAD_LE_4(buf), frv_inst);
  }
  return FRV_UNDEF; // instruction length > 32 bit
}

static void strlcat(char* restrict dst, const char* src, size_t size) {
  while (*dst && size)
    dst++, size--;
  while (*src && size > 1)
    *dst++ = *src++, size--;
  if (size)
    *dst = 0;
}

static char* frv_format_int(int32_t val, char buf[static 16]) {
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
    [FRV_LB] = "lb", [FRV_LH] = "lh", [FRV_LW] = "lw", [FRV_LD] = "ld",
    [FRV_LBU] = "lbu", [FRV_LHU] = "lhu", [FRV_LWU] = "lwu",
    [FRV_SB] = "sb", [FRV_SH] = "sh", [FRV_SW] = "sw", [FRV_SD] = "sd",

    [FRV_ADDI] = "addi", [FRV_SLLI] = "slli", [FRV_SLTI] = "slti",
    [FRV_SLTIU] = "sltiu", [FRV_XORI] = "xori", [FRV_SRAI] = "srai",
    [FRV_SRLI] = "srli", [FRV_ORI] = "ori", [FRV_ANDI] = "andi",
    [FRV_ADD] = "add", [FRV_SLL] = "sll", [FRV_SLT] = "slt",
    [FRV_SLTU] = "sltu", [FRV_XOR] = "xor", [FRV_SRL] = "srl", [FRV_OR] = "or",
    [FRV_AND] = "and", [FRV_SUB] = "sub", [FRV_SRA] = "sra",

    [FRV_FENCE] = "fence", [FRV_FENCEI] = "fencei",
    [FRV_AUIPC] = "auipc", [FRV_LUI] = "lui",
    [FRV_JAL] = "jal", [FRV_JALR] = "jalr",
    [FRV_BEQ] = "beq", [FRV_BNE] = "bne", [FRV_BLT] = "blt", [FRV_BGE] = "bge",
    [FRV_BLTU] = "bltu", [FRV_BGEU] = "bgeu",
    [FRV_ECALL] = "ecall",

    [FRV_ADDIW] = "addiw", [FRV_SLLIW] = "slliw", [FRV_SRAIW] = "sraiw",
    [FRV_SRLIW] = "srliw", [FRV_ADDW] = "addw", [FRV_SLLW] = "sllw",
    [FRV_SRLW] = "srlw", [FRV_SUBW] = "subw", [FRV_SRAW] = "sraw",

    [FRV_MUL] = "mul", [FRV_MULH] = "mulh", [FRV_MULHSU] = "mulhsu",
    [FRV_MULHU] = "mulhu", [FRV_DIV] = "div", [FRV_DIVU] = "divu",
    [FRV_REM] = "rem", [FRV_REMU] = "remu", [FRV_MULW] = "mulw",
    [FRV_DIVW] = "divw", [FRV_DIVUW] = "divuw", [FRV_REMW] = "remw",
    [FRV_REMUW] = "remuw",

    [FRV_LRW] = "lr.w", [FRV_SCW] = "sc.w",
    [FRV_LRD] = "lr.d", [FRV_SCD] = "sc.d",
    [FRV_AMOADDW] = "amoadd.w", [FRV_AMOADDD] = "amoadd.d",
    [FRV_AMOSWAPW] = "amoswap.w", [FRV_AMOSWAPD] = "amoswap.d",
    [FRV_AMOXORW] = "amoxor.w", [FRV_AMOXORD] = "amoxor.d",
    [FRV_AMOORW] = "amoor.w", [FRV_AMOORD] = "amoor.d",
    [FRV_AMOANDW] = "amoand.w", [FRV_AMOANDD] = "amoand.d",
    [FRV_AMOMINW] = "amomin.w", [FRV_AMOMIND] = "amomin.d",
    [FRV_AMOMAXW] = "amomax.w", [FRV_AMOMAXD] = "amomax.d",
    [FRV_AMOMINUW] = "amominu.w", [FRV_AMOMINUD] = "amominu.d",
    [FRV_AMOMAXUW] = "amomaxu.w", [FRV_AMOMAXUD] = "amomaxu.d",

    [FRV_CSRRW] = "csrrw", [FRV_CSRRWI] = "csrrwi",
    [FRV_CSRRS] = "csrrs", [FRV_CSRRSI] = "csrrsi",
    [FRV_CSRRC] = "csrrc", [FRV_CSRRCI] = "csrrci",

    [FRV_FLW] = "flw", [FRV_FSW] = "fsw",
    [FRV_FMVXW] = "fmv.x.w", [FRV_FMVWX] = "fmv.w.x", [FRV_FCLASSS] = "fclass.s",
    [FRV_FMADDS] = "fmadd.s", [FRV_FMSUBS] = "fmsub.s",
    [FRV_FNMSUBS] = "fnmsub.s", [FRV_FNMADDS] = "fnmadd.s",
    [FRV_FADDS] = "fadd.s", [FRV_FSUBS] = "fsub.s",
    [FRV_FMULS] = "fmul.s", [FRV_FDIVS] = "fdiv.s", [FRV_FSQRTS] = "fsqrt.s",
    [FRV_FSGNJS] = "fsgnj.s", [FRV_FSGNJNS] = "fsgnjn.s",
    [FRV_FSGNJXS] = "fsgnjx.s", [FRV_FMINS] = "fmin.s", [FRV_FMAXS] = "fmax.s",
    [FRV_FLES] = "fle.s", [FRV_FLTS] = "flt.s", [FRV_FEQS] = "feq.s",
    [FRV_FCVTWS] = "fcvt.w.s", [FRV_FCVTWUS] = "fcvt.wu.s",
    [FRV_FCVTLS] = "fcvt.l.s", [FRV_FCVTLUS] = "fcvt.lu.s",
    [FRV_FCVTSW] = "fcvt.s.w", [FRV_FCVTSWU] = "fcvt.s.wu",
    [FRV_FCVTSL] = "fcvt.s.l", [FRV_FCVTSLU] = "fcvt.s.lu",
    // RV32D/RV64D
    [FRV_FLD] = "fld", [FRV_FSD] = "fsd",
    [FRV_FMVXD] = "fmv.x.d", [FRV_FMVDX] = "fmv.d.x", [FRV_FCLASSD] = "fclass.d",
    [FRV_FMADDD] = "fmadd.d", [FRV_FMSUBD] = "fmsub.d",
    [FRV_FNMSUBD] = "fnmsub.d", [FRV_FNMADDD] = "fnmadd.d",
    [FRV_FADDD] = "fadd.d", [FRV_FSUBD] = "fsub.d",
    [FRV_FMULD] = "fmul.d", [FRV_FDIVD] = "fdiv.d", [FRV_FSQRTD] = "fsqrt.d",
    [FRV_FSGNJD] = "fsgnj.d", [FRV_FSGNJND] = "fsgnjn.d",
    [FRV_FSGNJXD] = "fsgnjx.d", [FRV_FMIND] = "fmin.d", [FRV_FMAXD] = "fmax.d",
    [FRV_FLED] = "fle.d", [FRV_FLTD] = "flt.d", [FRV_FEQD] = "feq.d",
    [FRV_FCVTSD] = "fcvt.s.d", [FRV_FCVTDS] = "fcvt.d.s",
    [FRV_FCVTWD] = "fcvt.w.d", [FRV_FCVTWUD] = "fcvt.wu.d",
    [FRV_FCVTLD] = "fcvt.l.d", [FRV_FCVTLUD] = "fcvt.lu.d",
    [FRV_FCVTDW] = "fcvt.d.w", [FRV_FCVTDWU] = "fcvt.d.wu",
    [FRV_FCVTDL] = "fcvt.d.l", [FRV_FCVTDLU] = "fcvt.d.lu",
};

void frv_format(const FrvInst* inst, size_t len, char* restrict buf) {
  char tmp[18];
  if (!len)
    return;
  buf[0] = 0;
  
  if (inst->mnem >= sizeof mnem_str / sizeof mnem_str[0] || !mnem_str[inst->mnem]) {
    strlcat(buf, "<invalid>", len);
    return;
  }
  strlcat(buf, mnem_str[inst->mnem], len);
  if (inst->rd != FRV_REG_INV) {
    char* fmt = frv_format_int(inst->rd, tmp + 2);
    *--fmt = 'r';
    *--fmt = ' ';
    strlcat(buf, fmt, len);
  }
  if (inst->rs1 != FRV_REG_INV) {
    char* fmt = frv_format_int(inst->rs1, tmp + 2);
    *--fmt = 'r';
    *--fmt = ' ';
    strlcat(buf, fmt, len);
  }
  if (inst->rs2 != FRV_REG_INV) {
    char* fmt = frv_format_int(inst->rs2, tmp + 2);
    *--fmt = 'r';
    *--fmt = ' ';
    strlcat(buf, fmt, len);
  }
  if (inst->rs3 != FRV_REG_INV) {
    char* fmt = frv_format_int(inst->rs3, tmp + 2);
    *--fmt = 'r';
    *--fmt = ' ';
    strlcat(buf, fmt, len);
  }
  if (inst->imm) {
    char* fmt = frv_format_int(inst->imm, tmp + 2);
    *--fmt = ' ';
    strlcat(buf, fmt, len);
  }
}



// 主函数，将指令格式字符串转换为 FrvInst 结构体
int frv_parse(FrvInst* inst, const char* fmt) {

    char tmp[18];



    return 0;
}