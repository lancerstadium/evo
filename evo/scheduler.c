
#include "evo.h"


// ==================================================================================== //
//                                  scheduler: sync
// ==================================================================================== //

static void scheduler_prerun_sync(scheduler_t* scd, graph_t* g) {
    /// TODO: pre run subgraph by device
}

static void scheduler_run_sync(scheduler_t* scd, graph_t* g) {
    /// TODO: run subgraph by device
}

static void scheduler_wait_sync(scheduler_t* scd, graph_t* g) {
    /// TODO: wait
}

static void scheduler_posrun_sync(scheduler_t* scd, graph_t* g) {
    /// TODO: post run subgraph by device
}

static scheduler_t sync_scheduler = {
    .name = "sync",
    .prerun = scheduler_prerun_sync,
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