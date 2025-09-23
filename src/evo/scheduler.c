#include <evo.h>
#include <evo/util/log.h>


// ==================================================================================== //
//                                  scheduler: sync
// ==================================================================================== //

static void scheduler_prerun_sync(scheduler_t* scd, graph_t* g) {
    /// TODO: pre run subgraph by device
    if(!scd || !g || g->is_sub) {
        LOG_WARN("Scheduler only prerun Parent graph");
    }
    for(int i = 0; i < vector_size(g->sub_vec); i++) {
        graph_t* sg = g->sub_vec[i];
        device_t* dev = sg->dev;
        int ret = dev->itf->prerun(dev, sg);
        if(ret != 0) {
            sg->status = GRAPH_STATUS_ERROR;
            LOG_ERR("Prerun subgraph(%d) on %s fail\n", sg->idx, dev->name);
            return;
        }
        sg->status = GRAPH_STATUS_READY;
    }
}

static void scheduler_step_sync(scheduler_t* scd, graph_t* g, int n) {
    /// TODO: run subgraph by device
    if(!scd || !g || g->is_sub) {
        LOG_WARN("Scheduler only run Parent graph");
    }
    for(int i = 0; i < vector_size(g->sub_vec); i++) {
        graph_t* sg = g->sub_vec[i];
        device_t* dev = sg->dev;
        int ret = dev->itf->step(dev, sg, n);
        if(ret != 0) {
            sg->status = GRAPH_STATUS_ERROR;
            LOG_ERR("Run subgraph(%d) on %s fail\n", sg->idx, dev->name);
            return;
        }
        sg->status = GRAPH_STATUS_READY;
    }
}


static void scheduler_run_sync(scheduler_t* scd, graph_t* g) {
    /// TODO: run subgraph by device
    if(!scd || !g || g->is_sub) {
        LOG_WARN("Scheduler only run Parent graph");
    }
    for(int i = 0; i < vector_size(g->sub_vec); i++) {
        graph_t* sg = g->sub_vec[i];
        device_t* dev = sg->dev;
        int ret = dev->itf->run(dev, sg);
        if(ret != 0) {
            sg->status = GRAPH_STATUS_ERROR;
            LOG_ERR("Run subgraph(%d) on %s fail\n", sg->idx, dev->name);
            return;
        }
        sg->status = GRAPH_STATUS_READY;
    }
}

static void scheduler_wait_sync(scheduler_t* scd, graph_t* g) {
    /// TODO: wait
}

static void scheduler_posrun_sync(scheduler_t* scd, graph_t* g) {
    /// TODO: post run subgraph by device
    if(!scd || !g || g->is_sub) {
        LOG_WARN("Scheduler only run Parent graph");
    }
    for(int i = 0; i < vector_size(g->sub_vec); i++) {
        graph_t* sg = g->sub_vec[i];
        device_t* dev = sg->dev;
        int ret = dev->itf->posrun(dev, sg);
        if(ret != 0) {
            sg->status = GRAPH_STATUS_ERROR;
            LOG_ERR("Posrun subgraph(%d) on %s fail\n", sg->idx, dev->name);
            return;
        }
        sg->status = GRAPH_STATUS_READY;
    }
}

static scheduler_t sync_scheduler = {
    .name = "sync",
    .prerun = scheduler_prerun_sync,
    .step = scheduler_step_sync,
    .run = scheduler_run_sync,
    .wait = scheduler_wait_sync,
    .posrun = scheduler_posrun_sync,
};

// ==================================================================================== //
//                                  scheduler API
// ==================================================================================== //

scheduler_t* scheduler_get_default() {
    return &sync_scheduler;
}