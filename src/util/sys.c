
#include <evo/util/sys.h>
#include <string.h>
#ifdef _MSC_VER
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/time.h>
#endif

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
#endif // CONFIG_ARCH_CORTEX_M

char* sys_get_file_ext(const char* path) {
    char* trg = strrchr(path, '.');
    return trg && *trg != '\0' ? (trg + 1) : NULL;
}
char* sys_get_file_name(const char* path) { 
    char* trg = strrchr(path, '/');
    return trg && *trg != '\0' ? (trg + 1) : NULL;
}


char* sys_memory_size(int size) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB", "EB"}; // 定义内存单位
    int unit_index = 0;  // 当前单位的索引
    double display_size = size;  // 将size转换为浮点数以便进行单位转换

    // 持续将字节数转换为更大的单位，直到找到合适的单位
    while (display_size >= 1024 && unit_index < 6) {
        display_size /= 1024;
        unit_index++;
    }

    // 分配用于存储结果的字符串
    char* result = (char*)malloc(20 * sizeof(char));  // 预留足够的空间
    if (result != NULL) {
        // 将结果格式化为字符串，保留两位小数
        snprintf(result, 20, "%.2f %s", display_size, units[unit_index]);
    }

    return result;  // 返回内存字符串
}

// ==================================================================================== //
//                                       system time
// ==================================================================================== //

double sys_time() {
#ifdef _MSC_VER
    LARGE_INTEGER freq;
    LARGE_INTEGER pc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&pc);
    return pc.QuadPart * 1000.0 / freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + (tv.tv_usec / 1000.0);
#endif
}