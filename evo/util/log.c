
#include "log.h"

#include <stdarg.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#include "lock.h"

static mutex_t log_locker;
static const char* map_table[] = {"EMERG", "ALERT", "CRIT", "ERROR", "WARN", "NOTICE", "INFO", "DEBUG"};

static void safety_log(struct logger* logger, char* message) {
    if (0 != message[LOG_MAX_MSG - 1]) {
        message[LOG_MAX_MSG - 1] = 0;
    }

    mutex_lock(&log_locker);
    logger->output_func(message);
    mutex_unlock(&log_locker);
}

static void do_log(struct logger* logger, enum log_level level, const char* fmt, ...) {
    if (logger->log_level < level || level > LOG_LEVEL_DEBUG) {
        return;
    }
#ifdef ANDROID
    va_list _ap;
    va_start(_ap, fmt);

    switch (level) {
        case LOG_EMERG:
        case LOG_ALERT:
        case LOG_CRIT: {
            __android_log_print(ANDROID_LOG_FATAL, "evo", fmt, _ap);
            break;
        }
        case LOG_ERR: {
            __android_log_print(ANDROID_LOG_ERROR, "evo", fmt, _ap);
            break;
        }
        case LOG_WARN: {
            __android_log_print(ANDROID_LOG_WARN, "evo", fmt, _ap);
            break;
        }
        case LOG_NOTICE:
        case LOG_INFO: {
            __android_log_print(ANDROID_LOG_INFO, "evo", fmt, _ap);
            break;
        }
        case LOG_DEBUG: {
            __android_log_print(ANDROID_LOG_DEBUG, "evo", fmt, _ap);
            break;
        }
        default: {
            __android_log_print(ANDROID_LOG_VERBOSE, "evo", fmt, _ap);
        }
    }
    va_end(_ap);

    return;

#else
    va_list ap;
    char msg[LOG_MAX_MSG] = {0};
    int max_len = LOG_MAX_MSG;
    int left = max_len;
    char* p = msg;
    int ret;

    if (logger->option.print_time) {
        time_t t = time(NULL);
        ret = strftime(p, left, "%Y-%m-%d %X ", localtime(&t));
        left -= ret;
        p += ret;
    }

    if (left <= 1) {
        return safety_log(logger, msg);
    }

    if (logger->option.print_level) {
        ret = snprintf(p, left, "%s ", map_table[level]);
        left -= ret;
        p += ret;
    }

    if (left <= 1) {
        return safety_log(logger, msg);
    }

    if (logger->option.print_prefix && logger->prefix) {
        ret = snprintf(p, left, "%s ", logger->prefix);
        left -= ret;
        p += ret;
    }

    if (left <= 1) {
        return safety_log(logger, msg);
    }

    va_start(ap, fmt);
    vsnprintf(p, left, fmt, ap);
    va_end(ap);

    return safety_log(logger, msg);
#endif
}

static void change_log_level(struct logger* logger, int level) {
    if (level < 0 || level > LOG_LEVEL_DEBUG) {
        return;
    }

    logger->log_level = level;
}

static void set_output_func(struct logger* logger, void (*func)(const char*)) {
    logger->output_func = func;
}

static void output_stderr(const char* msg) {
    fprintf(stderr, "%s", msg);
}

struct logger* logger_get_default(void) {
    static int inited = 0;
    static struct logger default_logger;

    if (inited)
        return &default_logger;
    else
        mutex_init(&log_locker);

    mutex_lock(&log_locker);
    if (!inited) {
        inited = 1;
        default_logger.prefix = NULL;
        default_logger.log_level = LOG_DFT_LEVEL;
        default_logger.output_func = output_stderr;
        default_logger.log = do_log;
        default_logger.set_log_level = change_log_level;
        default_logger.set_output_func = set_output_func;
        default_logger.option.print_prefix = 0;
        default_logger.option.print_time = 0;
        default_logger.option.print_level = 0;
    }

    mutex_unlock(&log_locker);
    return &default_logger;
}
