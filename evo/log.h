/**
 * =================================================================================== //
 * @file log.h
 * @author lancerstadium (lancerstadium@163.com)
 * @brief math header file
 * @version 0.1
 * @date 2024-07-03
 * @copyright Copyright (c) 2024
 * =================================================================================== //
 */

// ==================================================================================== //
//                                       evo/log.h
// ==================================================================================== //

#ifndef __EVO_LOG_H__
#define __EVO_LOG_H__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define LOG_MAX_MSG 256
#define LOG_DFT_LEVEL LOG_LEVEL_INFO

enum log_level {
    LOG_LEVEL_EMERG,
    LOG_LEVEL_ALERT,
    LOG_LEVEL_CRIT,
    LOG_LEVEL_ERR,
    LOG_LEVEL_WARN,
    LOG_LEVEL_NOTICE,
    LOG_LEVEL_INFO,
    LOG_LEVEL_DEBUG
};

struct log_option {
    int print_prefix;
    int print_time;
    int print_level;
};

struct logger {
    const char* prefix;
    int log_level;
    struct log_option option;
    void (*output_func)(const char*);
    void (*log)(struct logger*, enum log_level, const char* fmt, ...);
    void (*set_log_level)(struct logger*, int level);
    void (*set_output_func)(struct logger*, void (*func)(const char*));
};

struct logger* logger_get_default(void);

#define LOG_SET_OUTPUT(func)                          \
    do {                                              \
        struct logger* logger = logger_get_default(); \
        logger->set_output_func(logger, func);        \
    } while (0)

#define LOG_SET_LEVEL(level)                          \
    do {                                              \
        struct logger* logger = logger_get_default(); \
        logger->set_log_level(logger, level);         \
    } while (0)

#define LOG_SET_PRINT_TIME(val)                       \
    do {                                              \
        struct logger* logger = logger_get_default(); \
        logger->option.print_time = val;              \
    } while (0)

#define LOG_SET_PRINT_LEVEL(val)                      \
    do {                                              \
        struct logger* logger = logger_get_default(); \
        logger->option.print_level = val;             \
    } while (0)

#define LOG_SET_PRINT_PREFIX(val)                     \
    do {                                              \
        struct logger* logger = logger_get_default(); \
        logger->option.print_prefix = val;            \
    } while (0)

#define LOG_SET_PREFIX(val)                           \
    do {                                              \
        struct logger* logger = logger_get_default(); \
        logger->prefix = val;                         \
    } while (0)

#define LOG(level, fmt, ...)                            \
    do {                                                \
        struct logger* logger = logger_get_default();   \
        logger->log(logger, level, fmt, ##__VA_ARGS__); \
    } while (0)

#define LOG_EMERG(fmt, ...) LOG(LOG_LEVEL_EMERG, fmt, ##__VA_ARGS__)
#define LOG_ALERT(fmt, ...) LOG(LOG_LEVEL_ALERT, fmt, ##__VA_ARGS__)
#define LOG_CRIT(fmt, ...) LOG(LOG_LEVEL_CRIT, fmt, ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) LOG(LOG_LEVEL_ERR, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) LOG(LOG_LEVEL_WARN, fmt, ##__VA_ARGS__)
#define LOG_NOTICE(fmt, ...) LOG(LOG_LEVEL_NOTICE, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) LOG(LOG_LEVEL_INFO, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) LOG(LOG_LEVEL_DEBUG, fmt, ##__VA_ARGS__)

#define LOGX(level, fmt, ...)                 \
    LOG(level, "%s:%d ", __FILE__, __LINE__); \
    LOG(level, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_LOG_H__