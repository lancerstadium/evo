#include <evo/task.h>


// ==================================================================================== //
//                                    task: Exec                                      
// ==================================================================================== //

#define CPU(ctx) ((CPUState(ISE)*)((ctx)->cpu))

Task_fn_def(Exec);

void TaskCtx_OP_def(Exec, set_status) (TaskCtx(Exec) *ctx, int status) {
    Task_info(Exec, "CPU Status: %s -> %s", cpustatus_tbl2[CPU(ctx)->status] , cpustatus_tbl2[status]);
    CPU(ctx)->status = status;
}

void TaskCtx_OP_def(Exec, init) (TaskCtx(Exec) *ctx, Val* val) {
    if(ctx == NULL) {
        ctx = malloc(sizeof(TaskCtx(Exec)));
    }
    ctx->cpu = CPUState_init(ISE, 1024);
    Task_info(Exec, "CPU Init pc : %s", ValHex(CPU(ctx)->pc));
    Task_info(Exec, "CPU Mem size: %lu Byte", CPU(ctx)->mem->len);
    Task_info(Exec, "CPU Status  : %s" , cpustatus_tbl2[CPU(ctx)->status]);
    if(val && val->len > 0) {
        CPUState_set_mem(ISE, ctx->cpu, Val_new_u32(0), val, val->len);
        Task_info(Exec, "CPU Img Load: %s ..", ValHex(CPUState_get_mem(ISE, CPU(ctx), Val_new_u32(0), 8)));
    }
}
void TaskCtx_OP_def(Exec, run) (TaskCtx(Exec) *ctx) {
    size_t len = 4;
    TaskCtx_OP(Exec, execute) (ctx, len);
}

void TaskCtx_OP_def(Exec, rundbg) (TaskCtx(Exec) *ctx, UNUSED Val* val) {
    Task_ast(Exec, ctx != NULL, "Exec ctx is null");
    TaskCtx_OP(Exec, execute) (ctx, Val_as_u64(val, 0));
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
}

void TaskCtx_OP_def(Exec, execute) (TaskCtx(Exec) *ctx, size_t step) {
    for(size_t i = 0; i < step; i++){
        TaskCtx_OP(Exec, execone)(ctx, CPU(ctx)->pc);
    }
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

void TaskCtx_OP_def(Dump, rundbg) (TaskCtx(Dump) *ctx, UNUSED Val* val) {
    Task_ast(Dump, ctx != NULL, "Dump ctx is null");
    TaskCtx_OP(Dump, run)(ctx);
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