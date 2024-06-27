#ifndef EVO_TASK_H
#define EVO_TASK_H

#include <evo/evo.h>
#include <gen/gen.h>

// ==================================================================================== //
//                                    task: Load                                      
// ==================================================================================== //

Task_def(Load,

,

);

// ==================================================================================== //
//                                    task: Encode                                      
// ==================================================================================== //

Task_def(Encode,

,

);

// ==================================================================================== //
//                                    task: Decode                                      
// ==================================================================================== //

Task_def(Decode,
    Val pc;                 /* pc   : Program Counter       */
    Val snpc;               /* snpc : Static Next PC        */
    Val dnpc;               /* dnpc : Dynamic Next PC       */
#if defined(CFG_PERF_DECODE)
#endif
,
#if defined(CFG_SISA)
    Insn(CFG_SISA) * TaskCtx_OP_ISA_def(Decode, run, CFG_SISA) (TaskCtx(Decode) *ctx, Val* bc);
#endif
#if defined(CFG_IISA)
    Insn(CFG_IISA) * TaskCtx_OP_ISA_def(Decode, run, CFG_IISA) (TaskCtx(Decode) *ctx, Val* bc);
#endif
#if defined(CFG_TISA)
    Insn(CFG_TISA) * TaskCtx_OP_ISA_def(Decode, run, CFG_TISA) (TaskCtx(Decode) *ctx, Val* bc);
#endif
);


// ==================================================================================== //
//                                    task: Exec                                      
// ==================================================================================== //


Task_def(Exec,

,

);


// ==================================================================================== //
//                                    task: Trans                                      
// ==================================================================================== //

Task_def(Trans,

,

);


// ==================================================================================== //
//                                    task: Dump                                      
// ==================================================================================== //

Task_def(Dump,
    char* path;             /* path : Output File Path      */
    ELFDump *elf;           /* elf  : Elf Dump Context      */
#if defined(CFG_PERF_DUMP)

#endif
,
    void TaskCtx_OP_def(Dump, elf) (TaskCtx(Dump) *ctx, char* name);
    void TaskCtx_OP_def(Dump, clear) (TaskCtx(Dump) *ctx);
);



#endif  // EVO_TASK_H