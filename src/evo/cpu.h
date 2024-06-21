

#ifndef _EVO_CPU_H_
#define _EVO_CPU_H_

#include <evo/typ.h>


#define CPUState(T)             CONCAT(CPUState_, T)
#define CPUState_T(T, ...) \
    typedef struct {       \
        __VA_ARGS__        \
    } CPUState(T)

#define CPUState_def(T, ...) \
    CPUState_T(T, __VA_ARGS__ )



#endif // _EVO_CPU_H_