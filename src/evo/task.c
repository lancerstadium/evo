#include <evo/task.h>


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
//                                    task: Exec                                      
// ==================================================================================== //

Task_fn_def(Exec);

void TaskCtx_OP_def(Exec, init) (TaskCtx(Exec) *ctx, UNUSED Val* val) {
    if(ctx == NULL) {
        ctx = malloc(sizeof(TaskCtx(Exec)));
    }
    ctx->cpu = CPUState_init(ISE, 1024);
    ctx->snpc = Val_from(((CPUState(ISE)*)(ctx->cpu))->pc);
    Task_info(Exec, "CPU Init pc : %s", ValHex(((CPUState(ISE)*)(ctx->cpu))->pc));
    Task_info(Exec, "CPU Mem size: %lu Byte", ((CPUState(ISE)*)(ctx->cpu))->mem->len);
    Task_info(Exec, "CPU Status  : %s" , cpustatus_tbl2[((CPUState(ISE)*)(ctx->cpu))->status]);
    if(val && val->len > 0) {
        CPUState_set_mem(ISE, ctx->cpu, Val_new_u32(0), val);
        Task_info(Exec, "CPU Img Load: %s ..", ValHex(CPUState_get_mem(ISE, ctx->cpu, Val_new_u32(0), 8)));
    }
}
void TaskCtx_OP_def(Exec, run) (UNUSED TaskCtx(Exec) *ctx) {
    Task_ast(Exec, ctx != NULL, "Exec ctx is null");

}


// ==================================================================================== //
//                                    task: Dump                                      
// ==================================================================================== //

Task_fn_def(Dump);

void TaskCtx_OP_def(Dump, init) (TaskCtx(Dump) *ctx, Val* val) {
    if(ctx == NULL) {
        ctx = malloc(sizeof(TaskCtx(Dump)));
    }
    if(val && val->len > 0) {
        ctx->path = Val_as_str(val);
    } else {
        ctx->path = CFG_GEN_ELF;
    }
    ctx->elf = ELFDump_init();
}
void TaskCtx_OP_def(Dump, run) (TaskCtx(Dump) *ctx) {
    Task_ast(Dump, ctx != NULL, "Dump ctx is null");
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

#undef ISE