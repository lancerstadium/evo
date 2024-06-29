#include <evo/task.h>


// ==================================================================================== //
//                                    task: Disam                                      
// ==================================================================================== //





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
    ctx->cur_insn = NULL;
    ctx->cnt_insn = 0;
    ctx->e_sc = (CFG_PERF_TIMES == 1e3) ? "us" : ((CFG_PERF_TIMES == 1e6) ? "ms" : "s");
    ctx->e_s  = (struct timespec){0};
    ctx->e_e  = (struct timespec){0};
    ctx->e_tak = 0.0;
    ctx->e_tot = 0.0;
    if(val && val->len > 0) {
        CPUState_set_mem(ISE, ctx->cpu, Val_new_u64(CFG_MEM_BASE), val, val->len);
        Task_info(Exec, "CPU Img Load : %s ..", ValHex(CPUState_get_mem(ISE, CPU(ctx), Val_new_u64(CFG_MEM_BASE), 4)));
    }
    Task_info(Exec, "CPU Init ISA : %s", STR(ISE));
    Task_info(Exec, "CPU Init pc  : %s", ValAddr(CPU(ctx)->pc));
    Task_info(Exec, "CPU Mem size : %lu Byte", CPU(ctx)->mem->len);
    Task_info(Exec, "CPU Status   : %s" , cpustatus_tbl2[CPU(ctx)->status]);
}
void TaskCtx_OP_def(Exec, run) (TaskCtx(Exec) *ctx) {
    TaskCtx_OP(Exec, execute) (ctx, -1);
}

void TaskCtx_OP_def(Exec, rundbg) (TaskCtx(Exec) *ctx, UNUSED Val* val) {
    Task_ast(Exec, ctx != NULL, "Exec ctx is null");
    TaskCtx_OP(Exec, execute) (ctx, Val_as_i64(val, 0));
}

void TaskCtx_OP_def(Exec, execone) (TaskCtx(Exec) *ctx, Val* pc) {
    Task_ast(Exec, ctx != NULL, "Exec ctx is null");
    Val_copy(CPU(ctx)->pc, pc);
    Val_copy(CPU(ctx)->snpc, pc);

    /* ++++++++ EXECUTE PERF TIME PART ++++++++ */
    clock_gettime(CLOCK_MONOTONIC, &ctx->e_s);              // clock up
    /* ++++++++ EXECUTE PERF TIME PART ++++++++ */

    /* ======== EXECUTE INSN MAIN PART ======== */
    Val* bc = CPUState_fetch(ISE, CPU(ctx));                // fetch
    Insn(ISE) * insn = CPUState_decode(ISE, CPU(ctx), bc);  // decode
    CPUState_execute(ISE, CPU(ctx), insn);                  // execute
    /* ======== EXECUTE INSN MAIN PART ======== */

    /* ++++++++ EXECUTE PERF TIME PART ++++++++ */
    clock_gettime(CLOCK_MONOTONIC, &ctx->e_e);              // clock down
    ctx->e_tak = ((ctx->e_e.tv_sec - ctx->e_s.tv_sec) * 1e9 + ctx->e_e.tv_nsec - ctx->e_s.tv_nsec) / SOB_UT_TIMES;
    if (ctx->e_e.tv_nsec < ctx->e_s.tv_nsec) {              // wait 1 times                      
        ctx->e_tak += 1;                                                                          
    }                                                                     
    ctx->e_tot += ctx->e_tak;                               // Got total time
    /* ++++++++ EXECUTE PERF TIME PART ++++++++ */

    /* ++++++++ EXECUTE PERF CONUT PART +++++++ */
    ctx->cnt_insn++;                                        // Got total exec insns
    /* ++++++++ EXECUTE PERF COUNT PART +++++++ */

    ctx->cur_insn = (Insn(ISE) *)(insn);
    char insn_buf[48];
    Insn_display(RV, insn, insn_buf);
    // UnitTest_msg("%s", insn_buf);
    Task_info(Exec, "%-36s "_CYAN("%12.4f %2s"), insn_buf, ctx->e_tak, ctx->e_sc);
}

void TaskCtx_OP_def(Exec, execute) (TaskCtx(Exec) *ctx, int step) {
    Task_ast(Exec, ctx != NULL, "Exec ctx is null");
    switch (CPU(ctx)->status) {
        case CPU_END:
        case CPU_ABORT:
            Task_info(Exec, "Exec Task has ended, exit or run again");
            return;
        default: CPU(ctx)->status = CPU_RUN; break;
    }
    for(int i = 0; i < step || step < 0; i++){
        TaskCtx_OP(Exec, execone)(ctx, CPU(ctx)->pc);
        if(CPU(ctx)->status != CPU_RUN) {
            break;
        }
    }
    switch (CPU(ctx)->status) {
        case CPU_END:
        case CPU_ABORT:{
            if(CPU(ctx)->halt_ret == 0) {
                Task_info(Exec, "CPUState: %s  PC: 0x%lx  HIT: " _GREEN("GOOD TRAP"), cpustatus_tbl2[CPU(ctx)->status], Val_as_u64(CPU(ctx)->pc,0));
            } else {
                Task_info(Exec, "CPUState: %s  PC: 0x%lx  HIT: " _RED("BAD TRAP"), cpustatus_tbl2[CPU(ctx)->status], Val_as_u64(CPU(ctx)->pc,0));
            }
            char* info = TaskCtx_OP(Exec, execinfo)(ctx);
            Task_info(Exec, "\n%s", info);
            free(info);
            break;
        }
        case CPU_QUIT: {
            char* info = TaskCtx_OP(Exec, execinfo)(ctx);
            Task_info(Exec, "\n%s", info);
            free(info);
            break;
        }
        case CPU_RUN:
        default: CPU(ctx)->status = CPU_STOP; break;
    }
}

char* TaskCtx_OP_def(Exec, execinfo) (TaskCtx(Exec) *ctx) {
    char* res = malloc(512);
    Task_ast(Exec, ctx != NULL, "Exec ctx is null");
    Task_ast(Exec, res != NULL, "Res buffer is null");
    res[0] = '\0';
    snprintf(res, 512, 
        "┌───────────────────────────────────────────┐\n"
        "│ Total Insn Count  : %-12lu          │\n"
        "│ Total Exec Time   : %-12.4f %2s       │\n"
        "│ Emulate Frequency : %-12.4f insn/ms  │\n"
        "└───────────────────────────────────────────┘" 
        , ctx->cnt_insn, ctx->e_tot, ctx->e_sc, (double)(ctx->cnt_insn) / ctx->e_tot);
    return res;
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