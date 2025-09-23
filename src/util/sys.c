
#include <evo/util/sys.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)              // Windows platform: _MSC_VER
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)                            // Linux platform
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#elif defined(__APPLE__) && defined(__MACH__)       // macOS platform
#include <mach/mach.h>
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


char* sys_mem_size(size_t size) {
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

// ==================================================================================== //
//                                       system device
// ==================================================================================== //


uint64_t sys_mem_usage() {
    uint64_t memory_usage = 0;

#if defined(_WIN32) || defined(_WIN64)             // Windows
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        memory_usage = pmc.PrivateUsage;           // Memory usage in bytes
    }

#elif defined(__linux__)                           // Linux
    FILE* file = fopen("/proc/self/statm", "r");
    if (file) {
        long pages;
        if (fscanf(file, "%ld", &pages) == 1) {
            memory_usage = pages * (uint64_t)sysconf(_SC_PAGESIZE);  // Convert pages to bytes
        }
        fclose(file);
    }

#elif defined(__APPLE__) && defined(__MACH__)      // macOS
    struct task_basic_info info;
    mach_msg_type_number_t info_count = TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &info_count) == KERN_SUCCESS) {
        memory_usage = info.resident_size;         // Memory usage in bytes
    }

#endif
    return memory_usage;
}

uint64_t sys_cpu_cycle() {
    uint64_t cycles = 0;
#if defined(__riscv)                            // RISC-V
    asm volatile("rdcycle %0" : "=r"(cycles));
#elif defined(__x86_64__) || defined(_M_X64)    // x86_64
    unsigned int hi, lo;
    asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
    cycles = ((uint64_t)lo) | (((uint64_t)hi) << 32);
#elif defined(__aarch64__)                      // ARM
    asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
#else
    #error "Unsupported architecture"           // Unsupport
#endif
    return cycles;
}

uint64_t sys_inst_count() {
    uint64_t instructions = 0;
#if defined(__riscv)                          // RISC-V
    asm volatile("rdinstret %0" : "=r"(instructions));
#elif defined(__aarch64__)                    // ARM
    asm volatile("mrs %0, pmccntr_el0" : "=r"(instructions)); // ARMv8 PMU counter
#elif defined(__x86_64__) || defined(_M_X64)  // x86_64
    unsigned int lo, hi;
    asm volatile("rdpmc" : "=a"(lo), "=d"(hi) : "c"(0)); // Counter 0 for instructions
    instructions = ((uint64_t)hi << 32) | lo;
#else
    #error "Unsupported architecture"         // Unsupport
#endif
    return instructions;
}



int sys_has_avx() {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_avx_vnni() {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_avx2() {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_avx512() {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_avx512_vbmi() {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_avx512_vnni() {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_avx512_bf16() {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_fma() {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_neon() {
#if defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int sys_has_sve() {
#if defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int sys_has_arm_fma() {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int sys_has_riscv_v() {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int sys_has_metal() {
#if defined(EVO_ACC_METAL)
    return 1;
#else
    return 0;
#endif
}

int sys_has_f16c() {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_fp16_va() {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int sys_has_wasm_simd() {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_blas() {
#if defined(EVO_ACC_BLAS) || defined(EVO_ACC_CUDA) || defined(EVO_ACC_VULKAN) || defined(EVO_ACC_SYCL)
    return 1;
#else
    return 0;
#endif
}

int sys_has_cuda() {
#if defined(EVO_ACC_CUDA)
    return 1;
#else
    return 0;
#endif
}

int sys_has_vulkan() {
#if defined(EVO_ACC_VULKAN)
    return 1;
#else
    return 0;
#endif
}

int sys_has_kompute() {
#if defined(EVO_ACC_KOMPUTE)
    return 1;
#else
    return 0;
#endif
}

int sys_has_sycl() {
#if defined(EVO_ACC_SYCL)
    return 1;
#else
    return 0;
#endif
}

int sys_has_rpc() {
#if defined(EVO_ACC_RPC)
    return 1;
#else
    return 0;
#endif
}

int sys_has_cann() {
#if defined(EVO_ACC_CANN)
    return 1;
#else
    return 0;
#endif
}

int sys_has_llamafile() {
#if defined(EVO_ACC_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int sys_has_gpublas() {
    return sys_has_cuda() || sys_has_vulkan() || sys_has_kompute() || sys_has_sycl();
}

int sys_has_sse3() {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_ssse3() {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_vsx() {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int sys_has_matmul_int8() {
#if defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}