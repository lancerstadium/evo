

#ifndef UTIL_ALLOC_H
#define UTIL_ALLOC_H

#include "macro.h"
#include <stdlib.h>
#include <string.h>

#define ALLOC_CAPACITY_INIT_SIZE 16

#define ALLOC_DEC_TYPE(type) \
    typedef struct { \
        type *ptr; \
        u32 count; \
        u32 capacity; \
    } type##_alloc_t; \
    type##_alloc_t *type##_alloc_new(); \
    void type##_alloc_init(type##_alloc_t *alloc); \
    void type##_alloc_clear(type##_alloc_t *alloc); \
    void type##_alloc_realloc(type##_alloc_t *alloc, u32 capacity); \
    void type##_alloc_push(type##_alloc_t *alloc, type *ptr); \
    void type##_alloc_pushn(type##_alloc_t *alloc, type *ptr, u32 count); \
    type *type##_alloc_pop(type##_alloc_t *alloc); \
    type *type##_alloc_popn(type##_alloc_t *alloc, u32 count); \
    type *type##_alloc_get(type##_alloc_t *alloc, u32 index); \
    void type##_alloc_fill_write(type##_alloc_t *alloc, type *ptr);
    

#define ALLOC_DEF_METHOD(type) \
    void type##_alloc_init(type##_alloc_t *alloc) { \
        alloc->ptr = NULL; \
        alloc->count = 0; \
        alloc->capacity = ALLOC_CAPACITY_INIT_SIZE; \
    } \
    type##_alloc_t *type##_alloc_new() { \
        type##_alloc_t *alloc = (type##_alloc_t *)calloc(1, sizeof(type##_alloc_t)); \
        type##_alloc_init(alloc); \
        return alloc; \
    } \
    void type##_alloc_clear(type##_alloc_t *alloc) { \
        if (alloc->ptr) { \
            free(alloc->ptr); \
            alloc->ptr = NULL; \
        } \
        alloc->count = 0; \
        alloc->capacity = ALLOC_CAPACITY_INIT_SIZE; \
    } \
    void type##_alloc_realloc(type##_alloc_t *alloc, u32 capacity) { \
        if (alloc->ptr) { \
            alloc->ptr = realloc(alloc->ptr, capacity * sizeof(type)); \
            alloc->capacity = capacity; \
        } else { \
            alloc->ptr = calloc(capacity, sizeof(type)); \
        } \
        alloc->capacity = capacity; \
    } \
    void type##_alloc_push(type##_alloc_t *alloc, type *ptr) { \
        if (alloc->count >= alloc->capacity) { \
            type##_alloc_realloc(alloc, alloc->capacity * 2); \
        } \
        alloc->ptr[alloc->count++] = *ptr; \
    } \
    void type##_alloc_pushn(type##_alloc_t *alloc, type *ptr, u32 count) { \
        if (alloc->count + count >= alloc->capacity) { \
            type##_alloc_realloc(alloc, alloc->capacity * 2); \
        } \
        memcpy(alloc->ptr + alloc->count, ptr, count * sizeof(type)); \
        alloc->count += count; \
    } \
    type *type##_alloc_pop(type##_alloc_t *alloc) { \
        if (alloc->count == 0) { \
            return NULL; \
        } \
        return &alloc->ptr[--alloc->count]; \
    } \
    type *type##_alloc_popn(type##_alloc_t *alloc, u32 count) { \
        if (alloc->count < count) { \
            return NULL; \
        } \
        return &alloc->ptr[alloc->count -= count]; \
    } \
    type *type##_alloc_get(type##_alloc_t *alloc, u32 index) { \
        if (index >= alloc->count) { \
            return NULL; \
        } \
        return &alloc->ptr[index]; \
    } \
    void type##_alloc_fill_write(type##_alloc_t *alloc, type *ptr) { \
        type##_alloc_clear(alloc); \
        type##_alloc_pushn(alloc, ptr, 1); \
    }

#endif // UTIL_ALLOC_H