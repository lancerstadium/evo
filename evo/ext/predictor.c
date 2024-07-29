#include "../evo.h"
#include "../util/sys.h"
#include <math.h>

predictor_t* predictor_new(int n_pes, int pe_fp32s, int fp32_cycles, double frequency, double mem_bandwidth, double mem_concurrent_fp32,  double memory_efficiency, double l2_speed_frac, uint8_t is_max_mode, int batch_size, double launch_time) {
    predictor_t *pred = sys_malloc(sizeof(predictor_t));
    pred->n_pes = n_pes;
    pred->pe_fp32s = pe_fp32s;
    pred->fp32_cycles = fp32_cycles;
    pred->frequency = frequency;
    pred->mem_bandwidth = mem_bandwidth;
    pred->mem_fp32_bandwidth = mem_bandwidth / 4.0;
    pred->l2_fp32_bandwidth = pred->mem_fp32_bandwidth * l2_speed_frac;
    pred->memory_efficiency = memory_efficiency;
    pred->mem_concurrent_fp32 = mem_concurrent_fp32;
    pred->batch_size = batch_size;
    pred->launch_time = launch_time;
    pred->is_max_mode = is_max_mode;
    return pred;
}

double ceil_efficiency(double x, double parallel) {
    return x / (ceil(x / parallel) * parallel);
}

double mem_concurrent_efficiency(double n, double interval, double concurrent) {
    if (interval < 0) {
        interval = 0;
    }
    if (n > concurrent) {
        return ceil_efficiency(n, concurrent);
    }
    if (n + interval > concurrent) {
        return n / concurrent;
    }
    return n / (n + interval);
}

double pe_reduce_compute_latency(predictor_t *pred, int c_parallel, int n_elements) {
    int n = ceil(n_elements / 2.0);
    double latency = 0.0;

    while (n > 1) {
        double efficiency = ceil_efficiency(n * c_parallel, pred->pe_fp32s * pred->fp32_cycles);
        latency += (ceil(n * c_parallel / (double)pred->pe_fp32s) / efficiency) / pred->frequency;
        n = ceil(n / 2.0);
    }

    return latency * pred->batch_size;
}

double calc_memory_latency(predictor_t *pred, double all_pe_memory, double tot_fused_memory, double req_size, double req_interval) {
    double global_latency = tot_fused_memory / pred->mem_fp32_bandwidth;
    double l2_efficiency = mem_concurrent_efficiency(req_size, req_interval, pred->mem_concurrent_fp32);
    double l2_latency = all_pe_memory / pred->l2_fp32_bandwidth / l2_efficiency;
    return global_latency + l2_latency;
}
