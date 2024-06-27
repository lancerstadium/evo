#include <evo/task.h>


#if defined(CFG_MODE_ITP) || defined(CFG_MODE_AOT) || defined(CFG_MODE_JIT) || defined(CFG_MODE_HYB)
// typedef CONCAT(CPUState_, CFG_IISA) CPUState;
#define CSE CFG_IISA
#elif defined(CFG_MODE_EMU)
// typedef CONCAT(CPUState_, CFG_SISA) CPUState;
#define CSE CFG_SISA
#else
#error Unsupport EVO_MODE, Config options: EMU / ITP / AOT / JIT / HYB 
#endif


// ==================================================================================== //
//                                    task: Exec                                      
// ==================================================================================== //

Task_fn_def(Exec);

void TaskCtx_OP_def(Exec, init) (UNUSED TaskCtx(Exec) *ctx, UNUSED Val* val) {
    if(ctx == NULL) {
        ctx = malloc(sizeof(TaskCtx(Exec)));
    }
    ctx->cpu = CPUState_init(CSE, 56);
    ctx->snpc = Val_from(((CPUState(CSE)*)(ctx->cpu))->pc);
    Task_info(Exec, "CPU init: %s", ValHex(ctx->snpc));
}
void TaskCtx_OP_def(Exec, run) (UNUSED TaskCtx(Exec) *ctx) {

}


// ==================================================================================== //
//                                    task: Dump                                      
// ==================================================================================== //

Task_fn_def(Dump);

void TaskCtx_OP_def(Dump, init) (TaskCtx(Dump) *ctx, Val* val) {
    if(val && val->len > 0) {
        ctx->path = Val_as_str(val);
    } else {
        ctx->path = CFG_GEN_ELF;
    }
    ctx->elf = ELFDump_init();
}
void TaskCtx_OP_def(Dump, run) (TaskCtx(Dump) *ctx) {
    if (ctx->path && strcmp(ctx->path, "") != 0) {
        TaskCtx_OP(Dump, elf)(ctx, ctx->path);
    }
}

void TaskCtx_OP_def(Dump, elf) (TaskCtx(Dump) *ctx, char* name) {
    ELFDump_gen(ctx->elf, name);
    Task_info(Dump, _GREEN_BD("+") " %s", name);
}

void TaskCtx_OP_def(Dump, clear) (TaskCtx(Dump) *ctx) {
    ELFDump_free(ctx->elf);
}


// ==================================================================================== //
//                                    task: Undef                                     
// ==================================================================================== //

#undef CSE