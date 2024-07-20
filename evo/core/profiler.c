#include "../evo.h"
#include "../util/log.h"
#include "../util/sys.h"

static void profiler_exec_report(profiler_t *p) {
    if(!p || p->type != PROFILER_TYPE_EXEC) return;
    LOG_INFO("[Report: Exec]\n");
    LOG_INFO("| --------------------------------------------- |\n");
    LOG_INFO("|      Layer Name      |  Exec Time   | Proport |\n");
    LOG_INFO("| -------------------- | ------------ | ------- |\n");
    for(int i = 0; i < p->exec_nnode; i++) {
        LOG_INFO("| %20s | %9.3f ms | %6.2f% |\n", p->exec_node_vec[i]->name, p->exec_time_vec[i], p->exec_time_vec[i] /  p->exec_time_vec[p->exec_nnode] * 100);
    }
    LOG_INFO("| --------------------------------------------- |\n");
}

profiler_t * profiler_new(profiler_type_t type) {
    switch(type) {
        case PROFILER_TYPE_CUSTOM: {
            profiler_t* p = (profiler_t*)sys_malloc(sizeof(profiler_t));
            if(!p) return NULL;
            p->type = type;
            p->report = NULL;       /* Add report func by yourself  */
            p->custom = NULL;       /* Add data by yourself         */
            return p;
        }
        case PROFILER_TYPE_EXEC: {
            profiler_t* p = (profiler_t*)sys_malloc(sizeof(profiler_t));
            if(!p) return NULL;
            p->type = type;
            p->report = profiler_exec_report;
            p->exec_node_idx = 0;
            p->exec_nnode = 0;
            p->exec_node_vec = vector_create();
            p->exec_time_vec = vector_create();
            return p;
        }
        default: return NULL;
    }
}

void profiler_report(profiler_t *p) { if(p) p->report(p); }

void profiler_free(profiler_t *p) {
    if(!p) return;
    switch(p->type) {
        case PROFILER_TYPE_CUSTOM: {
            if(p) {
                if(p->custom) sys_free(p->custom);
                sys_free(p);
                p = NULL;
            }
            return;
        }
        case PROFILER_TYPE_EXEC: {
            if(p) {
                if(p->exec_node_vec) vector_free(p->exec_node_vec);
                if(p->exec_time_vec) vector_free(p->exec_time_vec);
                sys_free(p);
                p = NULL;
            }
            return;
        }
        default: {
            if(p) {
                sys_free(p);
                p = NULL;
            }
            return;
        } 
    }
}