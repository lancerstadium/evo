

#ifndef UTIL_HASHMAP_H
#define UTIL_HASHMAP_H

#include "gtype.h"

typedef struct {
    // Pointer to the value
    void* value;
    // The key name
    char key[];

} HashMapData;


typedef struct {
    HashMapData data;
    size_t size;
    size_t count;
} HashMap;

HashMap* hashmap_create(size_t size);
int hashmap_hash(HashMap* hashmap, const char* key);
void hashmap_insert(HashMap* hashmap, const char* key, void** data, size_t len);
void* hashmap_data(HashMap* hashmap, const char* key);




#endif // UTIL_HASHMAP_H