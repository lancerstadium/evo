


#ifndef _EVO_TASK_H_
#define _EVO_TASK_H_

#include <evo/typ.h>
#include <def/elf.h>

#define TaskCtx(T) CONCAT(TaskCtx_, T)
#define TaskCtx_OP(T, OP) CONCAT3(TaskCtx_, T##_, OP)
#define TaskCtx_OP_def(T, OP) UNUSED TaskCtx_OP(T, OP)
#define TaskCtx_T(T, S) \
    typedef struct {    \
        S               \
    } TaskCtx(T)

#define TaskCtx_def(T, S) \
    TaskCtx_T(T, S);

#define Task(T) CONCAT(Task_, T)
#define Task_OP(T, OP) CONCAT3(Task_, T##_, OP)
#define Task_OP_def(T, OP) UNUSED Task_OP(T, OP)
#define Task_T(T)            \
    typedef struct Task(T) { \
        const char* name;    \
        TaskCtx(T) ctx;      \
    } Task(T)

#define Task_def(T, S, ...)                        \
    TaskCtx_def(T, S);                             \
    Task_T(T);                                     \
    __VA_ARGS__                                    \
    Task(T) * Task_OP_def(T, create)(char* name) { \
        Task(T)* t = malloc(sizeof(Task(T)));      \
        t->name = name;                            \
        TaskCtx_OP(T, init)(&t->ctx);              \
        return t;                                  \
    }                                              \
    void Task_OP_def(T, run)(Task(T) * t) {        \
        TaskCtx_OP(T, run)(&t->ctx);               \
    }

#define Task_str(T) STR(T)
#define Task_create(T, name) Task_OP(T, create)(name)
#define Task_run(T, t) Task_OP(T, run)(t)

#include <def/elf.h>

Task_def(Dump,
    ElfCtx *elf;
,
    static inline void TaskCtx_OP_def(Dump, init) (TaskCtx(Dump) *ctx) {
        ctx->elf = ElfCtx_init();
    }
    static inline void TaskCtx_OP_def(Dump, elf) (TaskCtx(Dump) *ctx, char* name) {
        ElfCtx_gen(ctx->elf, name);
    }
    static inline void TaskCtx_OP_def(Dump, run) (TaskCtx(Dump) *ctx) {
        TaskCtx_OP(Dump, elf)(ctx, "a.out");
    }
    static inline void TaskCtx_OP_def(Dump, clean) (TaskCtx(Dump) *ctx) {
        ElfCtx_free(ctx->elf);
    }
);


#endif // _EVO_TASK_H_