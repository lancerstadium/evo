

#include <evo/evo.h>


#if defined(CFG_MODE_ITP) || defined(CFG_MODE_AOT) || defined(CFG_MODE_JIT) || defined(CFG_MODE_HYB)
typedef CONCAT(CPUState_, CFG_IISA) CPUState;
#elif defined(CFG_MODE_EMU)
typedef CONCAT(CPUState_, CFG_SISA) CPUState;
#else
#error Unsupport EVO_MODE, Config options: EMU / ITP / AOT / JIT / HYB 
#endif

Task_fn_def(Dump);

void TaskCtx_OP_def(Dump, init) (TaskCtx(Dump) *ctx) {
    ctx->elf = ElfCtx_init();
}
void TaskCtx_OP_def(Dump, elf) (TaskCtx(Dump) *ctx, char* name) {
    ElfCtx_gen(ctx->elf, name);
    Task_info(Dump, _GREEN_BD("+") " %s", name);
}
void TaskCtx_OP_def(Dump, run) (TaskCtx(Dump) *ctx) {
    TaskCtx_OP(Dump, elf)(ctx, CFG_GEN_ELF);
}
void TaskCtx_OP_def(Dump, clean) (TaskCtx(Dump) *ctx) {
    ElfCtx_free(ctx->elf);
}