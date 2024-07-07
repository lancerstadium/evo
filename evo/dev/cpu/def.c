#include "def.h"
#include "../../sys.h"


// ==================================================================================== //
//                                       cpu: graph info
// ==================================================================================== //

static cpu_graph_info_t* cpu_graph_info_new(graph_t *g) {
    cpu_graph_info_t* g_info = (cpu_graph_info_t*)sys_malloc(sizeof(cpu_graph_info_t));
    if(!g_info) return NULL;
    // Init info record vector
    g_info->exec_nnode = 0;
    g_info->exec_node_vec = vector_create();
    g_info->exec_time_vec = vector_create();
    // Update graph
    g->info = g_info;
    return g_info;
}

EVO_UNUSED static void cpu_graph_info_free(cpu_graph_info_t *g_info) {
    if(g_info) {
        if(g_info->exec_node_vec) vector_free(g_info->exec_node_vec);
        if(g_info->exec_time_vec) vector_free(g_info->exec_time_vec);
        sys_free(g_info);
        g_info = NULL;
    }
}

// ==================================================================================== //
//                                       cpu: interface
// ==================================================================================== //

static int cpu_init(device_t* dev) {
    if(dev) {
        dev->rsv = resolver_get_default();
        return 0;
    }
    return -1;
}

static int cpu_prerun(device_t *dev, graph_t *g) {
    /// TODO: Init exec graph info and load device mem?
    if(!dev || !g)
        return -1;
    cpu_graph_info_t* g_info = cpu_graph_info_new(g);
    /// TODO: Foreach Node in graph should be found
    for(int i = 0; i < g->nnode; i++) {
        node_t * nd = graph_get_node(g, g->nodes_vec[i]);
        if(nd) {
            op_t* trg_op = device_find_op(dev, nd->op->type);
            if(trg_op) {
                op_copy(nd->op, trg_op);
                vector_add(&(g_info->exec_node_vec), *nd);
                vector_add(&(g_info->exec_time_vec), 0.0);
                g_info->exec_nnode++;
            }
        }
    }
    return 0;
}

static int cpu_run(device_t *dev, graph_t *g) {
    /// TODO: Foreach Node in graph should trun
    return 0;
}

static int cpu_posrun(device_t *dev, graph_t *g) {
    
    /// TODO: Foreach Node in graph should postrun

    /// TODO: Release exec graph info
    return 0;
}

static int cpu_release(device_t* dev) {
    if(dev) {
        dev->rsv = NULL;
        return 0;
    }
    return -1;
}

// ==================================================================================== //
//                                       cpu: define
// ==================================================================================== //

static interface_t cpu_itf = {
    .init = cpu_init,
    .prerun = cpu_prerun,
    .run = cpu_run,
    .posrun = cpu_posrun,
    .release = cpu_release
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

// ==================================================================================== //
//                                       cpu: API
// ==================================================================================== //


device_t* device_get_cpu() {
    return &cpu_dev;
}