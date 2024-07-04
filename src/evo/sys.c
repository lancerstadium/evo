
#include "sys.h"
#include <string.h>

// ==================================================================================== //
//                                      sys port API
// ==================================================================================== //

void* sys_malloc(size_t size) {
    return malloc(size);
}

void sys_free(void* ptr) {
    return free(ptr);
}

void* sys_realloc(void* ptr, size_t size) {
    return realloc(ptr, size);
}


// ==================================================================================== //
//                                      string API
// ==================================================================================== //

#ifdef CONFIG_ARCH_CORTEX_M
char* sys_strdup(const char* src) {
    if (src == NULL)
        return NULL;
    int n = strlen(src);
    char* new_str = (char*)sys_malloc(n + 1);
    if (new_str == NULL)
        return NULL;
    memcpy(new_str, src, n + 1);
    return new_str;
}
#else

char* sys_strdup(const char* src) {
    return strdup(src);
}

#endif