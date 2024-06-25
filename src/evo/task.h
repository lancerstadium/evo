#ifndef EVO_TASK_H
#define EVO_TASK_H

#include <evo/evo.h>
#include <gen/gen.h>

Task_def(Load,

,

);


Task_def(Decode,
    Val pc;                 /* pc   : Program Counter       */
    Val snpc;               /* snpc : Static Next PC        */
    Val dnpc;               /* dnpc : Dynamic Next PC       */
#if defined(CFG_PERF_DECODE)
#endif
,
#if defined(CFG_SISA)
    Insn(CFG_SISA) TaskCtx_OP_ISA_def(Decode, run, CFG_SISA) (TaskCtx(Decode) *ctx, Val insn);
#endif
#if defined(CFG_IISA)
    Insn(CFG_IISA) TaskCtx_OP_ISA_def(Decode, run, CFG_IISA) (TaskCtx(Decode) *ctx, Val insn);
#endif
#if defined(CFG_TISA)
    Insn(CFG_TISA) TaskCtx_OP_ISA_def(Decode, run, CFG_TISA) (TaskCtx(Decode) *ctx, Val insn);
#endif
);

Task_def(Dump,
    ELFDump *elf;           /* elf  : Elf Dump Context      */
#if defined(CFG_PERF_DUMP)

#endif
,
    void TaskCtx_OP_def(Dump, init) (TaskCtx(Dump) *ctx);
    void TaskCtx_OP_def(Dump, elf) (TaskCtx(Dump) *ctx, char* name);
    void TaskCtx_OP_def(Dump, run) (TaskCtx(Dump) *ctx);
    void TaskCtx_OP_def(Dump, clear) (TaskCtx(Dump) *ctx);
);



#endif