#include "def.h"
#include <evo/util/sys.h>
#include <evo/util/log.h>


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
    if(!dev || !g)
        return -1;
    g->prof = profiler_new(PROFILER_TYPE_EXEC);
    if(!g->prof) {
        LOG_ERR("CPU Prerun Fail: No profiler!\n");
    }
    for(int i = 0; i < g->nnode; i++) {
        node_t * nd = graph_get_node(g, g->nodes_vec[i]);
        node_bind_op(nd);
        if(nd->op && nd->op->init) {
            vector_add(&(g->prof->exec_node_vec), nd);
            vector_add(&(g->prof->exec_time_vec), 0.0);
            g->prof->exec_nnode++;
        }
    }
    vector_add(&(g->prof->exec_time_vec), 0.0);     // Sum Time
    return 0;
}

static int cpu_step(device_t *dev, graph_t *g, int n) {
    if(!dev || !g) {
        LOG_ERR("CPU Step Fail: No device or graph!\n");
        return -1;
    }
    if(!g->prof) {
        LOG_ERR("CPU Step Fail: No device graph profiler!\n");
        return -1;
    }
    if(g->prof->exec_node_idx >= g->prof->exec_nnode) {
        LOG_WARN("CPU Step End: No more node to run!\n");
        return 0;
    }

    for(int i = 0; (i < n) && (g->prof->exec_node_idx < g->prof->exec_nnode); i++, g->prof->exec_node_idx++) {
        node_t* nd = g->prof->exec_node_vec[g->prof->exec_node_idx];
        if(!nd->op || !nd->op->init) {
            LOG_ERR("CPU Run Fail: Node %s no operator!\n", nd->name);
            return -1;
        }
        double time_st, time_ed;

        nd->op->init(nd);                           // --- Init Operator
        nd->op->reshape(nd);                        // --- Reshape Operator
        time_st = sys_time();                       // --- Clock up
        nd->op->forward(nd);                        // --- Forward Operator
        if(g->mode == 1 && nd->op->backward) {
            nd->op->backward(nd);                   // --- Backward Operator
        }
        time_ed = sys_time();                       // --- Clock down
        nd->op->exit(nd);                           // --- Exit Operator

        if(g->prof->exec_time_vec) {
            g->prof->exec_time_vec[g->prof->exec_node_idx] = time_ed - time_st;
            g->prof->exec_time_vec[g->prof->exec_nnode] += (time_ed - time_st);
            LOG_INFO("[RUN] Node: %s  Op: %s  Time: %f ms\n",nd->name, op_name(nd->op->type), g->prof->exec_time_vec[g->prof->exec_node_idx]);
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
    if(!g->prof) {
        LOG_ERR("CPU Run Fail: No device graph info!\n");
        return -1;
    }
    int start = (g->mode == 1) ? (g->prof->exec_nnode - 1) : 0;
    int end = (g->mode == 1) ? -1 : g->prof->exec_nnode;
    int step = (g->mode == 1) ? -1 : 1;
    double time_st, time_ed;

    for(int i = start; i != end; i += step) {
        g->prof->exec_node_idx = i;
        node_t* nd = g->prof->exec_node_vec[i];
        if(!nd->op || !nd->op->init) {
            LOG_ERR("CPU Run Fail: Node %s no operator!\n", nd->name);
            return -1;
        }
    
        nd->op->init(nd);                           // --- Init Operator
        nd->op->reshape(nd);                        // --- Reshape Operator
        time_st = sys_time();                       // --- Clock up
        nd->op->forward(nd);                        // --- Forward Operator
        if(g->mode == 1 && nd->op->backward) {
            nd->op->backward(nd);                   // --- Backward Operator
        }
        time_ed = sys_time();                       // --- Clock down
        nd->op->exit(nd);                           // --- Exit Operator
        
        if(g->prof->exec_time_vec) {
            g->prof->exec_time_vec[i] = time_ed - time_st;
            g->prof->exec_time_vec[g->prof->exec_nnode] += (time_ed - time_st);
        }
    }
    return 0;
}

static int cpu_posrun(device_t *dev, graph_t *g) {
    /// TODO: Foreach Node in graph should postrun
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
    sg->dev = internal_device_get_default();
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
    return internal_device_find("cpu");
}