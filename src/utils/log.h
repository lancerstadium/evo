

// ==================================================================================== //
//                                    utils: log
// ==================================================================================== //

#ifndef UTILS_LOG_H
#define UTILS_LOG_H

// ==================================================================================== //
//                                     Include
// ==================================================================================== //

#include "color.h"
#include <stdlib.h>

// ==================================================================================== //
//                                  Pub Data: log
// ==================================================================================== //

// log日志打印：日志等级
typedef enum {
    LOG_LEVEL_TRACE,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_FATAL,
    LOG_LEVEL_ASSERT_TRUE,
    LOG_LEVEL_ASSERT_FALSE
} LogLevel;

// ==================================================================================== //
//                                  Pri API: log
// ==================================================================================== //

void log_log(int level, const char *file, int line, const char *fmt, ...);


// ==================================================================================== //
//                                  Pub API: log
// ==================================================================================== //

#ifndef UTILS_LOG_DISABLE
// log日志打印：打印调试信息
#define log_trace(...) log_log(LOG_LEVEL_TRACE, __FILE__, __LINE__, __VA_ARGS__)

// log日志打印：打印调试信息
#define log_debug(...) log_log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __VA_ARGS__)

// log日志打印：打印信息
#define log_info(...) log_log(LOG_LEVEL_INFO, __FILE__, __LINE__, __VA_ARGS__)

// log日志打印：打印警告
#define log_warn(...) log_log(LOG_LEVEL_WARN, __FILE__, __LINE__, __VA_ARGS__)

// log日志打印：打印错误
#define log_error(...) log_log(LOG_LEVEL_ERROR, __FILE__, __LINE__, __VA_ARGS__)

// log日志打印：打印严重错误
#define log_fatal(...) \
    do{ \
        log_log(LOG_LEVEL_FATAL, __FILE__, __LINE__, __VA_ARGS__); \
        exit(-1); \
    } while(0)

// log日志打印：断言
#define log_assert(expr, ...) \
    do{ \
        if (expr) { \
            log_log(LOG_LEVEL_ASSERT_TRUE, __FILE__, __LINE__, ANSI_FMT(#expr, ANSI_BRIGHT_GREEN) " " __VA_ARGS__); \
            exit(-1); \
        } else { \
            log_log(LOG_LEVEL_ASSERT_FALSE, __FILE__, __LINE__, ANSI_FMT(#expr, ANSI_BRIGHT_RED) " " __VA_ARGS__); \
        } \
    } while(0)

#else
// log日志打印：不打印
#define log_trace(...)
#define log_debug(...)
#define log_info(...)
#define log_warn(...)
#define log_error(...)
#define log_fatal(...)
#define log_assert(expr, ...)
#endif


#endif // UTILS_LOG_H