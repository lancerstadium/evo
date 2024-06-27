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

#define CPU(ctx) ((CPUState(ISE)*)(ctx->cpu))

Task_fn_def(Exec);

void TaskCtx_OP_def(Exec, init) (TaskCtx(Exec) *ctx, Val* val) {
    if(ctx == NULL) {
        ctx = malloc(sizeof(TaskCtx(Exec)));
    }
    ctx->cpu = CPUState_init(ISE, 1024);
    Task_info(Exec, "CPU Init pc : %s", ValHex(CPU(ctx)->pc));
    Task_info(Exec, "CPU Mem size: %lu Byte", CPU(ctx)->mem->len);
    Task_info(Exec, "CPU Status  : %s" , cpustatus_tbl2[CPU(ctx)->status]);
    if(val && val->len > 0) {
        CPUState_set_mem(ISE, ctx->cpu, Val_new_u32(0), val);
        Task_info(Exec, "CPU Img Load: %s ..", ValHex(CPUState_get_mem(ISE, CPU(ctx), Val_new_u32(0), 8)));
    }
}
void TaskCtx_OP_def(Exec, run) (TaskCtx(Exec) *ctx) {
    size_t len = 4;
    for(size_t i = 0; i < len; i++){
        TaskCtx_OP(Exec, execone)(ctx, CPU(ctx)->pc);
    }
}

void TaskCtx_OP_def(Exec, execone) (TaskCtx(Exec) *ctx, Val* pc) {
    Task_ast(Exec, ctx != NULL, "Exec ctx is null");
    Val_copy(CPU(ctx)->pc, pc);
    Val_copy(CPU(ctx)->snpc, pc);
    Val* bc = CPUState_fetch(ISE, CPU(ctx));
    Insn(ISE) * insn = CPUState_decode(ISE, CPU(ctx), bc);
    CPUState_execute(ISE, CPU(ctx), insn);
    
    char insn_buf[48];
    Insn_display(RV, insn, insn_buf);
    UnitTest_msg("%s", insn_buf);
    // UnitTest_msg("pc  : %s", ValHex(CPU(ctx)->pc));
    // UnitTest_msg("snpc: %s", ValHex(CPU(ctx)->snpc));
    // UnitTest_msg("dnpc: %s", ValHex(CPU(ctx)->dnpc));
}

#undef CPU

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