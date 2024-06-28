#ifndef EVO_TASK_H
#define EVO_TASK_H

#include <evo/evo.h>
#include <gen/gen.h>

#if defined(CFG_MODE_ITP) || defined(CFG_MODE_AOT) || defined(CFG_MODE_JIT) || defined(CFG_MODE_HYB)
// typedef CONCAT(CPUState_, CFG_IISA) CPUState;
#define ISE CFG_IISA
#elif defined(CFG_MODE_EMU)
// typedef CONCAT(CPUState_, CFG_SISA) CPUState;
#define ISE CFG_SISA
#else
#error Unsupport EVO_MODE, Config options: EMU / ITP / AOT / JIT / HYB 
#endif


// ==================================================================================== //
//                                    task: Load                                      
// ==================================================================================== //

Task_def(Load,

,

);


// ==================================================================================== //
//                                    task: Exec                                      
// ==================================================================================== //


Task_def(Exec,
    void* cpu;                  /* cpu  : CPU State of Arch     */
,
    void TaskCtx_OP_def(Exec, set_status) (TaskCtx(Exec) *ctx, int status);
    void TaskCtx_OP_def(Exec, execone) (TaskCtx(Exec) *ctx, Val* pc);
    void TaskCtx_OP_def(Exec, execute) (TaskCtx(Exec) *ctx, size_t step);
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