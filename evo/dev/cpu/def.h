
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

typedef struct cpu_graph_info cpu_graph_info_t;

// ==================================================================================== //
//                                       cpu: graph info
// ==================================================================================== //

struct cpu_graph_info {
    int exec_node_idx;          /* Node Index of exec */
    int exec_nnode;             /* Node Number of exec */
    node_vec_t exec_node_vec;
    double * exec_time_vec;     /* 0..nnode-1 for node, nnode for sum */
};

// ==================================================================================== //
//                                       cpu API
// ==================================================================================== //

device_t* device_reg_cpu();



#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_DEV_CPU_DEF_H__
