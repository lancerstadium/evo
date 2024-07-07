
// ==================================================================================== //
//                                 evo/dev/cpu/def.h
// ==================================================================================== //

#ifndef __EVO_DEV_CPU_DEF_H__
#define __EVO_DEV_CPU_DEF_H__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ==================================================================================== //
//                                       include
// ==================================================================================== //

#include "../../evo.h"


// ==================================================================================== //
//                                       dev: cpu
// ==================================================================================== //

static interface_t cpu_itf = {
    
};

static allocator_t cpu_alc = {

};

static optimizer_t cpu_opt = {

};

static device_t cpu_dev = {
    .name = "cpu",
    .itf  = &cpu_itf,
    .alc  = &cpu_alc,
    .opt  = &cpu_opt,
    .scd  = NULL
};





#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_DEV_CPU_DEF_H__
