#include "def.h"
#include "../../util/sys.h"
#include "../../util/log.h"


// ==================================================================================== //
//                                       cpu: graph info
// ==================================================================================== //

static cpu_graph_info_t* cpu_graph_info_new(graph_t *g) {
    cpu_graph_info_t* g_info = (cpu_graph_info_t*)sys_malloc(sizeof(cpu_graph_info_t));
    if(!g_info) return NULL;
    // Init info record vector
    g_info->exec_node_idx = 0;
    g_info->exec_nnode = 0;
    g_info->exec_node_vec = vector_create();
    g_info->exec_time_vec = vector_create();
    // Update graph
    g->info = g_info;
    return g_info;
}

static cpu_graph_info_t* cpu_graph_info_get(graph_t *g) {
    if(g && g->info) {
        return (cpu_graph_info_t*)(g->info);
    }
    return NULL;
}

static void cpu_graph_info_free(cpu_graph_info_t *g_info) {
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
                nd->op = trg_op;
                vector_add(&(g_info->exec_node_vec), nd);
                vector_add(&(g_info->exec_time_vec), 0.0);
                g_info->exec_nnode++;
            }
        }
    }
    vector_add(&(g_info->exec_time_vec), 0.0); // Sum Time
    return 0;
}

static int cpu_step(device_t *dev, graph_t *g, int n) {
    if(!dev || !g) {
        LOG_ERR("CPU Step Fail: No device or graph!\n");
        return -1;
    }
    cpu_graph_info_t* g_info = cpu_graph_info_get(g);
    if(!g_info) {
        LOG_ERR("CPU Step Fail: No device graph info!\n");
        return -1;
    }
    if(g_info->exec_node_idx >= g_info->exec_nnode) {
        LOG_WARN("CPU Step End: No more node to run!\n");
        return 0;
    }
    for(int i = 0; (i < n) && (g_info->exec_node_idx < g_info->exec_nnode); i++, g_info->exec_node_idx++) {
        node_t* nd = g_info->exec_node_vec[g_info->exec_node_idx];
        if(!nd->op || !nd->op->run) {
            LOG_ERR("CPU Step Fail: Node %s no operator %s !\n", nd->name, op_name(nd->op->type) ? op_name(nd->op->type) : "");
            return -1;
        }
        // ==== Clock up ====== //
        double time_st, time_ed;
        time_st = sys_time();
        nd->op->run(nd);
        time_ed = sys_time();
        // ==== Clock down ==== //
        if(g_info->exec_time_vec) {
            g_info->exec_time_vec[g_info->exec_node_idx] = time_ed - time_st;
            g_info->exec_time_vec[g_info->exec_nnode] += (time_ed - time_st);
            LOG_INFO("[RUN] Node: %s  Op: %s  Time: %f ms\n",nd->name, op_name(nd->op->type), g_info->exec_time_vec[g_info->exec_node_idx]);
        }
    }
    return 0;
}

static int cpu_run(device_t *dev, graph_t *g) {
    /// TODO: Foreach Node in graph should run
    if(!dev || !g) {
        LOG_ERR("CPU Run Fail: No device or graph!\n");
        return -1;
    }
    cpu_graph_info_t* g_info = cpu_graph_info_get(g);
    if(!g_info) {
        LOG_ERR("CPU Run Fail: No device graph info!\n");
        return -1;
    }
    for(int i = 0; i < g_info->exec_nnode; i++) {
        g_info->exec_node_idx = i;
        node_t* nd = g_info->exec_node_vec[i];
        LOG_INFO("+ op_type: %s\n", op_name(nd->op->type));
        if(!nd->op || !nd->op->run) {
            LOG_ERR("CPU Run Fail: Node %s no operator %s !\n", nd->name, op_name(nd->op->type) ? op_name(nd->op->type) : "");
            return -1;
        }
        // ==== Clock up ====== //
        double time_st, time_ed;
        time_st = sys_time();
        nd->op->run(nd);
        time_ed = sys_time();
        // ==== Clock down ==== //
        if(g_info->exec_time_vec) {
            g_info->exec_time_vec[i] = time_ed - time_st;
            g_info->exec_time_vec[g_info->exec_nnode] += (time_ed - time_st);
        }
    }
    return 0;
}

static int cpu_posrun(device_t *dev, graph_t *g) {
    /// TODO: Foreach Node in graph should postrun
    /// TODO: Release exec graph info: for more mem
    cpu_graph_info_free((cpu_graph_info_t*)g->info);
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
//                                       cpu: allocator
// ==================================================================================== //

void cpu_alloc(device_t *dev, graph_t *sg) {
    (void)dev;
    (void)sg;
    return;
}

// ==================================================================================== //
//                                       cpu: optimizer
// ==================================================================================== //

void cpu_graph_spilte(graph_t *g) {
    graph_t *sg = graph_as_sub(g);
    sg->dev = device_registry_get_default();
    /// TODO: spilte graphs
}

// ==================================================================================== //
//                                       cpu: define
// ==================================================================================== //

static interface_t cpu_itf = {
    .init = cpu_init,
    .prerun = cpu_prerun,
    .step = cpu_step,
    .run = cpu_run,
    .posrun = cpu_posrun,
    .release = cpu_release
};

static allocator_t cpu_alc = {
    .alloc = cpu_alloc
};

static optimizer_t cpu_opt = {
    .graph_spilte = cpu_graph_spilte
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

device_t* device_reg_cpu() {
    device_reg_dev(&cpu_dev);
    return device_registry_find("cpu");
}