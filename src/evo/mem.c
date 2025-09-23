#include <evo.h>
#include <evo/util/log.h>
#include <evo/util/sys.h>
#include <string.h>
#include "../vis/svgenc.h"


// ==================================================================================== //
//                                  mem: define
// ==================================================================================== //

#define MEM_ARENA_SVG_BUF_SIZE 1024 * 1024

// ==================================================================================== //
//                                  mem: blob
// ==================================================================================== //

mem_blob_t* mem_blob_new(char* name) {
    mem_blob_t* mb = malloc(sizeof(mem_blob_t));
    mb->name = name ? sys_strdup(name) : sys_strdup("blob");
    mb->idx = 0;
    mb->size = 0;
    mb->lower = 0;
    mb->upper = 0;
    mb->offset = -1;
    mb->arena = NULL;
    return mb;
}

void mem_blob_apply(mem_blob_t* mb, int64_t size, int64_t lower, int64_t upper, int64_t offset) {
    if(mb) {
        mb->size = size;
        mb->lower = lower;
        mb->upper = upper;
        mb->offset = offset;
    }
}

int64_t mem_blob_area(mem_blob_t mb) {
    return mb.size * (mb.upper - mb.lower); 
}

bool mem_blob_equal(mem_blob_t mb1, mem_blob_t mb2) {
    return (strcmp(mb1.name, mb2.name) == 0)
        && (mb1.idx == mb2.idx)
        && (mb1.size == mb2.size)
        && (mb1.lower == mb2.lower)
        && (mb1.upper == mb2.upper)
        && (mb1.offset == mb2.offset)
        && ((void*)(mb1.arena) == (void*)(mb2.arena));
}

void mem_blob_free(mem_blob_t* mb) {
    if(mb) {
        if(mb->name) sys_free(mb->name);
        mb->name = NULL;
        mb->arena = NULL;
        free(mb);
    }
    mb = NULL;
}


// ==================================================================================== //
//                                  mem: block
// ==================================================================================== //

static const char* mem_arena_type_table[MEM_ARENA_TYPE_LAST] = {
    [MEM_ARENA_TYPE_FREE]   = "Free",
    [MEM_ARENA_TYPE_NAIVE]  = "Naive",
    [MEM_ARENA_TYPE_LSTF]   = "Large Size Tenosr First",
    [MEM_ARENA_TYPE_SLTF]   = "Short Lifetime Tensor First"
};

mem_arena_t* mem_arena_new(char* name) {
    mem_arena_t* ma = malloc(sizeof(mem_arena_t));
    ma->name = name ? sys_strdup((const char*)name) : sys_strdup("arena");
    ma->type = MEM_ARENA_TYPE_FREE;
    ma->capacity = 0;
    ma->blobs = vector_create();
    return ma;
}

mem_arena_t* mem_arena_from_graph(graph_t* g) {
    if(g && g->nodes) {
        mem_arena_t* ma = mem_arena_new(g->name);
        node_t* nd1, *nd2;
        tensor_t* ts1, *ts2;
        int64_t size, lower, upper;
        mem_blob_t* blob;
        for(int i = 0; i < g->nnode; i++) {
            // Get lower
            nd1 = g->nodes[i];
            lower = i;
            for(int m = 0; m < nd1->nout; m++) {
                ts1 = nd1->out[m];
                if(ts1->is_param == 0) {
                    // Get Size
                    size = 1;
                    upper = lower;
                    for(int k = 0; k < ts1->ndim; k++) {
                        size *= ts1->dims[k];
                    }
                    // Search upper
                    for(int j = i + 1; j < g->nnode; j++) {
                        nd2 = g->nodes[j];
                        if(nd2) {
                            for(int n = 0; n < nd2->nin; n++) {
                                ts2 = nd2->in[n];
                                if(ts2 && ts2->is_param == 0 && ((void*)ts1 == (void*)ts2)) {   // Tenosr Equal
                                    upper = j;
                                }
                            }
                        }
                    }
                    if(lower < upper) {
                        blob = mem_blob_new(ts1->name);
                        mem_blob_apply(blob, size, lower, upper, 0);
                        mem_arena_push(ma, blob);
                    }
                }
            }
        }
        return ma;
    }
    return NULL;
}


void mem_arena_push(mem_arena_t* ma, mem_blob_t* mb) {
    if(ma && mb && ma->blobs) {
        mb->idx = vector_size(ma->blobs);
        mb->arena = ma;
        vector_add(&(ma->blobs), mb);
    }
}

void mem_arena_dump(mem_arena_t* ma) {
    if(ma && ma->blobs) {
        size_t tot = vector_size(ma->blobs);
        LOG_INFO("----------------------------------------------------------------------------------------------------\n");
        LOG_INFO(" %8s\t%16s\t%8s\t%8s\t%12s\t%12s\n", "index", "name", "lower", "upper", "size", "offset");
        for(size_t i = 0; i < tot; i++) {
            mem_blob_t* mb = ma->blobs[i];
            if(mb) {
                LOG_INFO(" %8ld\t%16s\t%8ld\t%8ld\t%12ld\t%12ld\n", mb->idx, mb->name, mb->lower, mb->upper, mb->size, mb->offset);
            }
        }
        LOG_INFO("----------------------------------------------------------------------------------------------------\n");
        LOG_INFO(" + Solve : %s\n", mem_arena_type_table[ma->type]);
        LOG_INFO(" + Size  : %ld\n", ma->capacity);
        LOG_INFO(" + Mem   : %s\n", sys_mem_size(ma->capacity));
        LOG_INFO("----------------------------------------------------------------------------------------------------\n");
    }
}

// 比较函数：首先按生命周期起始时间升序排序，若相同则按大小降序排序
int compare_blob(const void* a, const void* b) {
    mem_blob_t* blob_a = *(mem_blob_t**)a;
    mem_blob_t* blob_b = *(mem_blob_t**)b;

    if (blob_a->lower != blob_b->lower) {
        return (blob_a->lower - blob_b->lower);
    } else {
        return (blob_b->size - blob_a->size); // 大小降序
    }
}


void mem_arena_solve(mem_arena_t* ma, mem_arena_type_t type) {
    if(ma && ma->type != type) {
        switch(type) {
            case MEM_ARENA_TYPE_NAIVE: {    // Alloc All Mem
                if(ma->blobs) {
                    size_t tot = vector_size(ma->blobs);
                    
                    int64_t ofs = 0;
                    mem_blob_t* mb;
                    for(size_t i = 0; i < tot; i++) {
                        mb = ma->blobs[i];
                        mb->offset = ofs;
                        ofs += mb->size;
                    }
                    ma->capacity = ofs;
                }
                break;
            }
            case MEM_ARENA_TYPE_LSTF: {     // Large Size Tensor First
                if (ma->blobs) {
                    size_t tot = vector_size(ma->blobs);

                    // 初始化所有 blob 的 offset 为 -1（未分配）
                    for (size_t i = 0; i < tot; i++) {
                        ma->blobs[i]->offset = -1;
                    }

                    // Step 1: 按照生命周期起始时间升序排序，若相同则按大小降序
                    qsort(ma->blobs, tot, sizeof(mem_blob_t*), compare_blob);

                    // Step 2: 分配 offset，确保生命周期不冲突
                    int64_t ofs = 0;
                    for (size_t i = 0; i < tot; i++) {
                        mem_blob_t* mb = ma->blobs[i];
                        int64_t candidate_offset = 0;

                        // 从0开始尝试分配offset，确保生命周期无重叠
                        while (1) {
                            int conflict = 0;

                            // 检查已分配的 blobs，确保生命周期不冲突
                            for (size_t j = 0; j < i; j++) {
                                mem_blob_t* other_blob = ma->blobs[j];

                                // 确保生命周期无重叠，且 candidate_offset 不与已分配内存冲突
                                if (!(other_blob->upper < mb->lower || other_blob->lower > mb->upper) &&
                                    candidate_offset < other_blob->offset + other_blob->size &&
                                    candidate_offset + mb->size > other_blob->offset) {
                                    // 若冲突，移动 candidate_offset 到其他张量的结束位置
                                    candidate_offset = other_blob->offset + other_blob->size;
                                    conflict = 1;
                                    break;
                                }
                            }

                            // 无冲突，分配 candidate_offset
                            if (!conflict) break;
                        }

                        // 分配 offset 并更新 ofs
                        mb->offset = candidate_offset;
                        ofs = candidate_offset + mb->size > ofs ? candidate_offset + mb->size : ofs;
                    }

                    // 更新 arena 的 capacity
                    ma->capacity = ofs;
                }
                break;
            }
            default: break;
        }
        ma->type = type;
    }
}

void mem_arena_free(mem_arena_t* ma) {
    if(ma) {
        if(ma->name) sys_free(ma->name);
        ma->name = NULL;
        if(ma->blobs) vector_free(ma->blobs);
        ma->blobs = NULL;
        free(ma);
    }
    ma = NULL;
}

void mem_arena_save_svg(mem_arena_t* ma, const char* path, float ratio) {
    if(!ma || !path) return;
    // 0. Open file & Init Buf
    FILE *fptr = fopen(path, "w");
    if (!fptr) {
        LOG_ERR("Svg open %s fail!\n", path);
        return;
    }
    char* svg_buf = malloc(MEM_ARENA_SVG_BUF_SIZE * sizeof(char));
    // 1. Draw svg
    int w = 50;
    int h = 20;
    int n = vector_size(ma->blobs);
    int upper = 0;
    for(int i = 0; i < n; i++) {
        if(ma->blobs[i]->upper > upper)
            upper = ma->blobs[i]->upper;
    }
    float pad = 2;
    ratio = ratio > 0 ? ratio : 1024;
    int width = upper * w;                          // Life
    int height = (ma->capacity / ratio) * h;        // KB
    int padsize = 18;
    int invsize = 2;
    svg_header(svg_buf, width + 2 * padsize * pad, height + 2 * padsize * pad);
    svg_clip_region(svg_buf, padsize * pad, padsize * pad, width, height, "plot");
    mem_blob_t* mb;
    float xx, yy, ww, hh;
    for(int i = 0; i < n; i++) {
        mb = ma->blobs[i];
        xx = ((float)mb->lower) * w + padsize * pad;
        ww = ((float)mb->upper - mb->lower) * w - pad * 2;
        hh = ((float)mb->size / ratio) * h - pad * 2;
        yy = (float)height - ((float)mb->offset / ratio) * h - (hh <= 0 ? 1 : hh) + padsize * pad;
        svg_rectangle(svg_buf, xx, yy, ww, hh <= 0 ? 1 : hh, "grey", "black", 1, "plot");
    }
    svg_line(svg_buf, (padsize - invsize) * pad, height + (padsize + invsize) * pad, width + padsize * pad, height + (padsize + invsize) * pad, "black", 2, NULL);
    svg_line(svg_buf, (padsize - invsize) * pad, (padsize - invsize) * pad, (padsize - invsize) * pad, height + (padsize + invsize) * pad, "black", 2, NULL);
    char mem_buf[64];
    mem_buf[0] = '\0';
    char ooo_buf[64];
    ooo_buf[0] = '\0';
    sprintf(mem_buf, "lifetime: %d", upper);
    svg_text_bold(svg_buf, (width / 2), height + (padsize + invsize + 10) * pad, SVG_TXT_MIDDLE, mem_buf, NULL);
    sprintf(ooo_buf, "rotate(270, %.2f, %.2f) translate(0, 10)", (padsize - invsize - 8) * pad, (float)(height / 2));
    char* mem = sys_mem_size(ma->capacity);
    sprintf(mem_buf, "memory: %s", mem);
    free(mem);
    svg_text_transform(svg_buf, (padsize - invsize - 8) * pad, ((float)height / 2), SVG_TXT_MIDDLE, SVG_TXT_BOLD, ooo_buf, mem_buf, NULL);
    svg_footer(svg_buf);
    // -1. Output to file & Close file & Free svg buffer
    fprintf(fptr, "%s", svg_buf);
    fclose(fptr);
    free(svg_buf);
}