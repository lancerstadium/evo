#include "../evo.h"
#include "../util/log.h"
#include "../util/sys.h"
#include <string.h>

char* str_from_back(char* str, char sym, int back_idx) {
    int len = strlen(str);
    int count = 0;
    int index = -1;
    for (int i = len - 1; i >= 0; i--) {
        if (str[i] == sym) {
            count++;
            if (count == back_idx) {
                index = i + 1;
                break;
            }
        }
    }
    if (index != -1) {
        return &str[index];
    } else {
        return str;
    }
}

static void profiler_exec_report0(profiler_t* p) {
    if (!p || p->type != PROFILER_TYPE_EXEC) return;
    LOG_INFO("[Report: Exec]\n");
    LOG_INFO("| ----------------------------------------------- |\n");
    LOG_INFO("|       Layers(%3d)      |  Exec Time   | Proport |\n", p->exec_nnode);
    LOG_INFO("| ---------------------- | ------------ | ------- |\n");
    for (int i = 0; i < p->exec_nnode; i++) {
        char* name1 = str_from_back(p->exec_node_vec[i]->name, '_', 2);
        char* name2 = str_from_back(p->exec_node_vec[i]->name, '/', 1);
        char* cname = name1 > name2 ? name1 : name2;
        LOG_INFO("| %22s | %9.3f ms | %6.2f% |\n", cname, p->exec_time_vec[i], p->exec_time_vec[i] / p->exec_time_vec[p->exec_nnode] * 100);
    }
    LOG_INFO("| ---------------------- | ------------ | ------- |\n");
    LOG_INFO("| %22s | %9.3f ms | %6.2f% |\n", "Exec Total", p->exec_time_vec[p->exec_nnode], 100.0);
    LOG_INFO("| ----------------------------------------------- |\n");
}

static void profiler_exec_report1(profiler_t* p) {
    if (!p || p->type != PROFILER_TYPE_EXEC) return;
    op_type_t* exec_type_operator_vec = vector_create();
    int* exec_num_operator_vec = vector_create();
    double* exec_time_operator_vec = vector_create();
    int find_idx;
    for(int i = 0; i < p->exec_nnode; i++) {
        find_idx = -1;
        for(int j = 0; j < vector_size(exec_type_operator_vec); j++) {
            if(p->exec_node_vec[i]->op->type == exec_type_operator_vec[j]) {
                find_idx = j;
                break;
            }
        }
        if(find_idx >= 0) {
            exec_time_operator_vec[find_idx] += p->exec_time_vec[i];
            exec_num_operator_vec[find_idx] += 1;
        } else {
            vector_add(&exec_type_operator_vec, p->exec_node_vec[i]->op->type);
            vector_add(&exec_num_operator_vec, 1);
            vector_add(&exec_time_operator_vec, p->exec_time_vec[i]);
        }
    }
    LOG_INFO("[Report: Exec-Operator]\n");
    LOG_INFO("| ------------------------------------------------------- |\n");
    LOG_INFO("|      Operator(%3d)     |  num  |  Exec Time   | Proport |\n", vector_size(exec_type_operator_vec));
    LOG_INFO("| ---------------------- | ----- | ------------ | ------- |\n");
    for (int i = 0; i < vector_size(exec_type_operator_vec); i++) {
        LOG_INFO("| %22s | %5d | %9.3f ms | %6.2f% |\n", op_name(exec_type_operator_vec[i]), exec_num_operator_vec[i], exec_time_operator_vec[i], exec_time_operator_vec[i] / p->exec_time_vec[p->exec_nnode] * 100);
    }
    LOG_INFO("| ---------------------- | ----- | ------------ | ------- |\n");
    LOG_INFO("| %22s | %5d | %9.3f ms | %6.2f% |\n", "Exec Total", p->exec_nnode, p->exec_time_vec[p->exec_nnode], 100.0);
    LOG_INFO("| ------------------------------------------------------- |\n");
    vector_free(exec_time_operator_vec);
    vector_free(exec_num_operator_vec);
    vector_free(exec_type_operator_vec);
}

static void profiler_exec_report(profiler_t* p, int l) {
    switch(l) {
        case 1: profiler_exec_report1(p); break;
        case 0:
        default: profiler_exec_report0(p); break;
    }
}

profiler_t* profiler_new(profiler_type_t type) {
    switch (type) {
        case PROFILER_TYPE_CUSTOM: {
            profiler_t* p = (profiler_t*)sys_malloc(sizeof(profiler_t));
            if (!p) return NULL;
            p->type = type;
            p->report = NULL; /* Add report func by yourself  */
            p->custom = NULL; /* Add data by yourself         */
            return p;
        }
        case PROFILER_TYPE_EXEC: {
            profiler_t* p = (profiler_t*)malloc(sizeof(profiler_t));
            if (!p) return NULL;
            p->type = type;
            p->report = profiler_exec_report;
            p->exec_node_idx = 0;
            p->exec_nnode = 0;
            p->exec_time_vec = vector_create();
            p->exec_node_vec = vector_create();
            return p;
        }
        default:
            return NULL;
    }
}

void profiler_report(profiler_t* p, int l) {
    if (p && p->report) p->report(p, l);
}

void profiler_free(profiler_t* p) {
    if (!p) return;
    switch (p->type) {
        case PROFILER_TYPE_CUSTOM: {
            if (p) {
                if (p->custom) sys_free(p->custom);
                sys_free(p);
                p = NULL;
            }
            return;
        }
        case PROFILER_TYPE_EXEC: {
            if (p) {
                if (p->exec_node_vec) vector_free(p->exec_node_vec);
                if (p->exec_time_vec) vector_free(p->exec_time_vec);
                sys_free(p);
                p = NULL;
            }
            return;
        }
        default: {
            if (p) {
                sys_free(p);
                p = NULL;
            }
            return;
        }
    }
}