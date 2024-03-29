

// ==================================================================================== //
//                                    utils: log
// ==================================================================================== //

#ifndef UTIL_LOG_H
#define UTIL_LOG_H

// ==================================================================================== //
//                                     Include
// ==================================================================================== //

#include "color.h"
#include <stdlib.h>


// ==================================================================================== //
//                                     Define
// ==================================================================================== //

#define UTIL_LOG_ENABLE
// #define LOG_TRACE_INFO
#define LOG_FUNC_INFO
#define LOG_DEBUG_INFO
// #define LOG_DATA_INFO
// #define LOG_ASSERT_INFO

#define LOG_INIT_SIZE 8


// ==================================================================================== //
//                                  Pub Enum: LogLevel
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

typedef struct {
    const char* file;
} logger;

// logger* logger_get(const char* file, const char* func);
// void logger_kill(logger* logger);
// void logger_free();


// ==================================================================================== //
//                                  Pri API: log
// ==================================================================================== //

void log_msg(int level, const char *file, int line, const char *fmt, ...);


// ==================================================================================== //
//                                  Pub API: log
// ==================================================================================== //

#ifdef UTIL_LOG_ENABLE

#ifdef LOG_TRACE_INFO
// log日志打印：打印调试信息
#define log_trace(...) log_msg(LOG_LEVEL_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#else
#define log_trace(...)
#endif

// log日志打印：打印调试信息
#ifdef LOG_DEBUG_INFO
#define log_debug(...) log_msg(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#else
#define log_debug(...)
#endif

// log日志打印：打印信息
#define log_info(...) log_msg(LOG_LEVEL_INFO, __FILE__, __LINE__, __VA_ARGS__)

// log日志打印：打印警告
#define log_warn(...) log_msg(LOG_LEVEL_WARN, __FILE__, __LINE__, __VA_ARGS__)

// log日志打印：打印错误
#define log_error(...) log_msg(LOG_LEVEL_ERROR, __FILE__, __LINE__, __VA_ARGS__)

#define log_error_if(expr, ...) \
    do { \
        if((expr)) { \
            log_msg(LOG_LEVEL_ERROR, __FILE__, __LINE__, ANSI_FMT(#expr, ANSI_BRIGHT_RED) " "  __VA_ARGS__); \
            exit(-1); \
        } \
    } while(0)

// log日志打印：打印严重错误
#define log_fatal(...) \
    do{ \
        log_msg(LOG_LEVEL_FATAL, __FILE__, __LINE__, __VA_ARGS__); \
        exit(-1); \
    } while(0)


#ifdef LOG_ASSERT_INFO
#define log_assert_true(expr, ...) log_msg(LOG_LEVEL_ASSERT_TRUE, __FILE__, __LINE__, ANSI_FMT(#expr, ANSI_BRIGHT_GREEN) " " __VA_ARGS__);
#else
#define log_assert_true(expr, ...)
#endif

// log日志打印：断言
#define log_assert(expr, ...) \
    do{ \
        if (!(expr)) { \
            log_msg(LOG_LEVEL_ASSERT_FALSE, __FILE__, __LINE__, ANSI_FMT(#expr, ANSI_BRIGHT_RED) " " __VA_ARGS__); \
            exit(-1); \
        } else { \
            log_assert_true(expr, __VA_ARGS__) \
        } \
    } while(0)

#ifdef LOG_FUNC_INFO
#define LOG_TAG log_info(ANSI_FMT("%s", ANSI_BRIGHT_BLUE) "() is called", __func__);
#else
#define LOG_TAG
#endif

#else
// log日志打印：不打印
#define log_trace(...)
#define log_debug(...)
#define log_info(...)
#define log_warn(...)
#define log_error(...)
#define log_fatal(...)
#define log_assert(expr, ...)
#define LOG_TAG
#endif


#endif // UTILS_LOG_H