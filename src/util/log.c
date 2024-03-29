
#include "log.h"
#include <stdarg.h>
#include <time.h>
#include <stdio.h>

// ==================================================================================== //
//                                 Private Data: log
// ==================================================================================== //


// log日志打印：日志等级字符串
static const char* log_level_str[] = {"TRAC", "DEBU", "INFO", "WARN", "ERRO", "FATA", "ASSE", "ASSE"};
// log日志打印：日志等级颜色
static const char* log_level_color[] = {ANSI_BRIGHT_BLUE, ANSI_CYAN, ANSI_BRIGHT_GREEN, ANSI_BRIGHT_YELLOW, ANSI_BRIGHT_RED, ANSI_MAGENTA, ANSI_BRIGHT_GREEN, ANSI_BRIGHT_RED};



// ==================================================================================== //
//                                 Private Func: log
// ==================================================================================== //

// log日志打印：打印原型
void log_msg(int level, const char *file, int line, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    // 输出到错误信息里
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);


    fprintf(stderr, "["
#ifdef LOG_DATA_INFO
    "%04d-%02d-%02d "
#endif
    "%02d:%02d:%02d] ", 
#ifdef LOG_DATA_INFO
    tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday, 
#endif
    tm->tm_hour, tm->tm_min, tm->tm_sec);
    fprintf(stderr, "%s", log_level_color[level]);
    fprintf(stderr, "%4s " ANSI_RESET, log_level_str[level]);
    fprintf(stderr, ANSI_FMT("%s:%d: ", ANSI_BLACK), file, line);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr,"\n");
    va_end(ap);
}

