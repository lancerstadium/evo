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
//                                    task: Exec                                      
// ==================================================================================== //


Task_def(Exec,
    void* cpu;                  /* cpu  : CPU State of Arch     */
    Val* snpc;                  /* snpc : Static Next PC        */
    Val* dnpc;                  /* dnpc : Dynamic Next PC       */
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