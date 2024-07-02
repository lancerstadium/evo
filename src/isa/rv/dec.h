
#ifndef _ISA_RV_DEC_H_
#define _ISA_RV_DEC_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum RvOptions {
  RV_RV32 = 0 << 0,
  RV_RV64 = 1 << 0,
  RV_RV128 = 2 << 0,
  RV_RVMSK = 3 << 0,
};
typedef enum RvOptions RvOptions;

enum {
  RV_UNDEF = -1,
  RV_PARTIAL = -2,
};

enum {
  RV_REG_INV = (uint8_t) -1,
};

enum {
  RV_INVALID = 0,
  // RV32I
  RV_LB, RV_LH, RV_LW, RV_LD, RV_LBU, RV_LHU, RV_LWU,
  RV_SB, RV_SH, RV_SW, RV_SD,
  RV_ADDI, RV_SLLI, RV_SLTI, RV_SLTIU, RV_XORI, RV_SRAI, RV_SRLI, RV_ORI, RV_ANDI,
  RV_ADD, RV_SLL, RV_SLT, RV_SLTU, RV_XOR, RV_SRL, RV_OR, RV_AND, RV_SUB, RV_SRA,
  RV_FENCE, RV_FENCEI,
  RV_AUIPC, RV_LUI,
  RV_JAL, RV_JALR,
  RV_BEQ, RV_BNE, RV_BLT, RV_BGE, RV_BLTU, RV_BGEU,
  RV_ECALL,
  // RV64I
  RV_ADDIW, RV_SLLIW, RV_SRAIW, RV_SRLIW,
  RV_ADDW, RV_SLLW, RV_SRLW, RV_SUBW, RV_SRAW,

  // RV32M, RV64M
  RV_MUL, RV_MULH, RV_MULHSU, RV_MULHU, RV_DIV, RV_DIVU, RV_REM, RV_REMU,
  RV_MULW, RV_DIVW, RV_DIVUW, RV_REMW, RV_REMUW,

  // RV32A/RV64A
  RV_LRW, RV_SCW, RV_LRD, RV_SCD,
  RV_AMOADDW, RV_AMOSWAPW, RV_AMOXORW, RV_AMOORW, RV_AMOANDW,
  RV_AMOMINW, RV_AMOMAXW, RV_AMOMINUW, RV_AMOMAXUW,
  RV_AMOADDD, RV_AMOSWAPD, RV_AMOXORD, RV_AMOORD, RV_AMOANDD,
  RV_AMOMIND, RV_AMOMAXD, RV_AMOMINUD, RV_AMOMAXUD,

  // RV32/RV64 Zicsr
  RV_CSRRW, RV_CSRRS, RV_CSRRC, RV_CSRRWI, RV_CSRRSI, RV_CSRRCI,

  // RV32F/RV64F
  RV_FLW, RV_FSW, RV_FMVXW, RV_FMVWX, RV_FCLASSS,
  RV_FMADDS, RV_FMSUBS, RV_FNMSUBS, RV_FNMADDS,
  RV_FADDS, RV_FSUBS, RV_FMULS, RV_FDIVS, RV_FSQRTS,
  RV_FSGNJS, RV_FSGNJNS, RV_FSGNJXS, RV_FMINS, RV_FMAXS,
  RV_FLES, RV_FLTS, RV_FEQS,
  RV_FCVTWS, RV_FCVTWUS, RV_FCVTLS, RV_FCVTLUS,
  RV_FCVTSW, RV_FCVTSWU, RV_FCVTSL, RV_FCVTSLU,
  // RV32D/RV64D
  RV_FLD, RV_FSD, RV_FMVXD, RV_FMVDX, RV_FCLASSD,
  RV_FMADDD, RV_FMSUBD, RV_FNMSUBD, RV_FNMADDD,
  RV_FADDD, RV_FSUBD, RV_FMULD, RV_FDIVD, RV_FSQRTD,
  RV_FSGNJD, RV_FSGNJND, RV_FSGNJXD, RV_FMIND, RV_FMAXD,
  RV_FLED, RV_FLTD, RV_FEQD,
  RV_FCVTSD, RV_FCVTDS,
  RV_FCVTWD, RV_FCVTWUD, RV_FCVTLD, RV_FCVTLUD,
  RV_FCVTDW, RV_FCVTDWU, RV_FCVTDL, RV_FCVTDLU,
};

typedef struct RvInst RvInst;
// Note: structure layout is unstable.
struct RvInst {
  uint16_t mnem;
  uint8_t rd;
  uint8_t rs1;
  uint8_t rs2;
  uint8_t rs3;
  uint8_t misc;
  int32_t imm;
};

int rv_decode(size_t bufsz, const uint8_t* buf, RvOptions, RvInst* rv_inst);

// Note: actual format is unstable.
void rv_format(const RvInst* inst, size_t len, char* buf);

#ifdef __cplusplus
}
#endif

#endif  // _ISA_RV_DEC_H_

