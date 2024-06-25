/**
 * @file sob.h
 * @author lancerstadium (lancerstadium@163.com)
 * @brief Super No Build Toolkit
 * @version 0.0.4
 * @date 2024-06-13
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _SOB_SOB_H_
#define _SOB_SOB_H_


// ==================================================================================== //
//                                       Include
// ==================================================================================== //

#ifndef _WIN32

#define _POSIX_C_SOURCE 200809L
#define PATH_SEP "/"
#define PATH_SEPC '/'

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>

typedef pid_t SobPid;
typedef int SobFd;

#define ERRON_MSG (errno == 0 ? "" : strerror(errno))

#else

#define WIN32_MEAN_AND_LEAN
#define PATH_SEP "\\"
#define PATH_SEPC '\\'
#include "windows.h"
#include <process.h>

typedef HANDLE SobPid;
typedef HANDLE SobFd;

struct dirent {
    char d_name[MAX_PATH+1];
};

typedef struct DIR DIR;
DIR *opendir(const char *dirpath);
struct dirent *readdir(DIR *dirp);
int closedir(DIR *dirp);
LPSTR GetLastErrorAsString(void);

#define ERRON_MSG (GetLastErrorAsString())

#endif  // _WIN32


#include <time.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>
#include <assert.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdbool.h>


// ==================================================================================== //
//                                    sob: SOB Config (SOB)
// ==================================================================================== //

#ifdef __cplusplus
extern "C" {
#endif

// #define SOB_APP_OFF
// #define SOB_CLR_OFF
// #define SOB_LOG_DBG_OFF
#define SOB_DS_DSIZE 16
// 1e3: us, 1e6: ms, 1e9: s
#define SOB_UT_TIMES 1e6
// 1: Open assert message, 0: No assert message
#define SOB_UT_ASTMSG 0
// Max sub command number
#define SOB_AP_MSCMD 12
// Show flag numbers
#define SOB_AP_NFLAG 3
#define SOB_AP_LFLAG "--"
#define SOB_AP_SFLAG "-"
#define SOB_AP_GLCMD "all"



// ==================================================================================== //
//                                    sob: Typedef
// ==================================================================================== //




// ==================================================================================== //
//                                    sob: Func (FN)
// ==================================================================================== //


#if defined(__GNUC__) || defined(__clang__)
#define UNUSED      __attribute__((unused))
#define EXPORT      __attribute__((visibility("default")))
#define NORETURN    __attribute__((noreturn))
#define PACKED(D)   D __attribute__((packed))
#elif defined(MSC_VER)
#define UNUSED      __pragma(warning(suppress:4100))
#define EXPORT      __pragma(warning(suppress:4091))
#define PACKED(D)   __pragma(pack(push, 1)) D __pragma(pack(pop))
#define NORETURN
#else
#define UNUSED
#define EXPORT
#define PACKED(D)   D
#define NORETURN
#endif

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define REP1(M, a1) M (a1)
#define REP2(M, a1, a2) M (a1) REP_SEP M (a2)
#define REP3(M, a1, a2, a3) REP2 (M, a1, a2) REP_SEP M (a3)
#define REP4(M, a1, a2, a3, a4) REP3 (M, a1, a2, a3) REP_SEP M (a4)
#define REP5(M, a1, a2, a3, a4, a5) REP4 (M, a1, a2, a3, a4) REP_SEP M (a5)
#define REP6(M, a1, a2, a3, a4, a5, a6) REP5 (M, a1, a2, a3, a4, a5) REP_SEP M (a6)
#define REP7(M, a1, a2, a3, a4, a5, a6, a7) REP6 (M, a1, a2, a3, a4, a5, a6) REP_SEP M (a7)
#define REP8(M, a1, a2, a3, a4, a5, a6, a7, a8) REP7 (M, a1, a2, a3, a4, a5, a6, a7) REP_SEP M (a8)
#define REP9(M, a1, a2, a3, a4, a5, a6, a7, a8, a9) REP8 (M, a1, a2, a3, a4, a5, a6, a7, a8) REP_SEP M (a9)
#define REP_SEP ,

#define _STR(s) #s
#define STR(s) _STR(s)
#define _CONCAT(a, b) a ## b
#define CONCAT(a, b) _CONCAT(a, b)
#define CONCAT3(a, b, c) CONCAT(CONCAT(a, b), c)
#define CONCAT4(a, b, c, d) CONCAT(CONCAT3(a, b, c), d)
#define STR_BOOL(b) ((b) ? "true" : "false")
#define STR_FMT(SD, fmt, ...) sprintf(SD, fmt, __VA_ARGS__)
#define STR_FMTN(SD, N, fmt, ...) snprintf(SD, (size_t)(N), fmt, __VA_ARGS__)

#define STRLEN(cs) (sizeof(cs) - 1)
#define ARRLEN(arr) (sizeof(arr) / sizeof(arr[0]))

#define ROUNDUP(a, sz)   ((((uintptr_t)a) + (sz) - 1) & ~((sz) - 1))
#define ROUNDDOWN(a, sz) ((((uintptr_t)a)) & ~((sz) - 1))

#define BITMASK(bits)   ((1ull << (bits)) - 1)
#define BITS(x, hi, lo) (((x) >> (lo)) & BITMASK((hi) - (lo) + 1)) // similar to x[hi:lo] in verilog
#define EBYTE(v, s)     ((v >> (s * 8)) & 0xFF)

// ==================================================================================== //
//                                    sob: Macro Testing
// ==================================================================================== //

// macro testing
// See https://stackoverflow.com/questions/26099745/test-if-preprocessor-symbol-is-defined-inside-macro
#define CHOOSE2nd(a, b, ...) b
#define MUX_WITH_COMMA(contain_comma, a, b) CHOOSE2nd(contain_comma a, b)
#define MUX_MACRO_PROPERTY(p, macro, a, b) MUX_WITH_COMMA(concat(p, macro), a, b)
// define placeholders for some property
#define __P_DEF_0  X,
#define __P_DEF_1  X,
#define __P_ONE_1  X,
#define __P_ZERO_0 X,
// define some selection functions based on the properties of BOOLEAN macro
#define MUXDEF(macro, X, Y)  MUX_MACRO_PROPERTY(__P_DEF_, macro, X, Y)
#define MUXNDEF(macro, X, Y) MUX_MACRO_PROPERTY(__P_DEF_, macro, Y, X)
#define MUXONE(macro, X, Y)  MUX_MACRO_PROPERTY(__P_ONE_, macro, X, Y)
#define MUXZERO(macro, X, Y) MUX_MACRO_PROPERTY(__P_ZERO_,macro, X, Y)

// test if a boolean macro is defined
#define ISDEF(macro) MUXDEF(macro, 1, 0)
// test if a boolean macro is undefined
#define ISNDEF(macro) MUXNDEF(macro, 1, 0)
// test if a boolean macro is defined to 1
#define ISONE(macro) MUXONE(macro, 1, 0)
// test if a boolean macro is defined to 0
#define ISZERO(macro) MUXZERO(macro, 1, 0)
// test if a macro of ANY type is defined
// NOTE1: it ONLY works inside a function, since it calls `strcmp()`
// NOTE2: macros defined to themselves (#define A A) will get wrong results
#define isdef(macro) (strcmp("" #macro, "" str(macro)) != 0)

// simplification for conditional compilation
#define __IGNORE(...)
#define __KEEP(...) __VA_ARGS__
// keep the code if a boolean macro is defined
#define IFDEF(macro, ...) MUXDEF(macro, __KEEP, __IGNORE)(__VA_ARGS__)
// keep the code if a boolean macro is undefined
#define IFNDEF(macro, ...) MUXNDEF(macro, __KEEP, __IGNORE)(__VA_ARGS__)
// keep the code if a boolean macro is defined to 1
#define IFONE(macro, ...) MUXONE(macro, __KEEP, __IGNORE)(__VA_ARGS__)
// keep the code if a boolean macro is defined to 0
#define IFZERO(macro, ...) MUXZERO(macro, __KEEP, __IGNORE)(__VA_ARGS__)

// ==================================================================================== //
//                                    sob: Args (VA)
// ==================================================================================== //

#define VA_NARG(args...)            VA_NARG_(0, ##args, VA_RSEQ_N())
#define VA_NARG_(args...)           VA_ARG_N(args)
#define VA_ARG_FIRST(first, ...)    first
#define VA_ARG_REST(first, ...)     , ## __VA_ARGS__
#define VA_ARG_N(_0, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, N, ...) N
#define VA_RSEQ_N() 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
#define VA_FIRST(args...)           VA_FIRST_(args)
#define VA_FIRST_(F, ...)           F
#define VA_FIRST_STR(args...)       VA_FIRST_STR_(args)
#define VA_FIRST_STR_(F, ...)       #F
#define VA_REST_(F, args...)        args
#define VA_REST(args...)            VA_REST_(args)


// ==================================================================================== //
//                                    sob: Color (CL)
// ==================================================================================== //

#ifdef SOB_CLR_OFF

#define ANSI_RESET                  ""
#define ANSI_BOLD                   ""
#define ANSI_DIM                    ""
#define ANSI_ITALIC                 ""
#define ANSI_UNDERLINE              ""
#define ANSI_BLINK                  ""
#define ANSI_INVERT                 ""
#define ANSI_REVERSE                ""
#define ANSI_HIDDEN                 ""
#define ANSI_STRIKETHROUGH          ""

#define ANSI_FG_BLACK               ""
#define ANSI_FG_RED                 ""
#define ANSI_FG_GREEN               ""
#define ANSI_FG_YELLOW              ""
#define ANSI_FG_BLUE                ""
#define ANSI_FG_MAGENTA             ""
#define ANSI_FG_CYAN                ""
#define ANSI_FG_WHITE               ""
#define ANSI_FG_DEFAULT             ""

#define ANSI_BG_BLACK               ""
#define ANSI_BG_RED                 ""
#define ANSI_BG_GREEN               ""
#define ANSI_BG_YELLOW              ""
#define ANSI_BG_BLUE                ""
#define ANSI_BG_MAGENTA             ""
#define ANSI_BG_CYAN                ""
#define ANSI_BG_WHITE               ""
#define ANSI_BG_DEFAULT             ""

#define ANSI_FGB_BLACK              ""
#define ANSI_FGB_RED                ""
#define ANSI_FGB_GREEN              ""
#define ANSI_FGB_YELLOW             ""
#define ANSI_FGB_BLUE               ""
#define ANSI_FGB_MAGENTA            ""
#define ANSI_FGB_CYAN               ""
#define ANSI_FGB_WHITE              ""
#define ANSI_FGB_DEFAULT            ""
#else

#define ANSI_RESET                  "\x1b[0m"
#define ANSI_BOLD                   "\x1b[1m"
#define ANSI_DIM                    "\x1b[2m"
#define ANSI_ITALIC                 "\x1b[3m"
#define ANSI_UNDERLINE              "\x1b[4m"
#define ANSI_BLINK                  "\x1b[5m"
#define ANSI_INVERT                 "\x1b[6m"
#define ANSI_REVERSE                "\x1b[7m"
#define ANSI_HIDDEN                 "\x1b[8m"
#define ANSI_STRIKETHROUGH          "\x1b[9m"

#define ANSI_FG_BLACK               "\x1b[30m"
#define ANSI_FG_RED                 "\x1b[31m"
#define ANSI_FG_GREEN               "\x1b[32m"
#define ANSI_FG_YELLOW              "\x1b[33m"
#define ANSI_FG_BLUE                "\x1b[34m"
#define ANSI_FG_MAGENTA             "\x1b[35m"
#define ANSI_FG_CYAN                "\x1b[36m"
#define ANSI_FG_WHITE               "\x1b[37m"
#define ANSI_FG_DEFAULT             "\x1b[39m"

#define ANSI_BG_BLACK               "\x1b[40m"
#define ANSI_BG_RED                 "\x1b[41m"
#define ANSI_BG_GREEN               "\x1b[42m"
#define ANSI_BG_YELLOW              "\x1b[43m"
#define ANSI_BG_BLUE                "\x1b[44m"
#define ANSI_BG_MAGENTA             "\x1b[45m"
#define ANSI_BG_CYAN                "\x1b[46m"
#define ANSI_BG_WHITE               "\x1b[47m"
#define ANSI_BG_DEFAULT             "\x1b[49m"

#define ANSI_FGB_BLACK              "\x1b[90m"
#define ANSI_FGB_RED                "\x1b[91m"
#define ANSI_FGB_GREEN              "\x1b[92m"
#define ANSI_FGB_YELLOW             "\x1b[93m"
#define ANSI_FGB_BLUE               "\x1b[94m"
#define ANSI_FGB_MAGENTA            "\x1b[95m"
#define ANSI_FGB_CYAN               "\x1b[96m"
#define ANSI_FGB_WHITE              "\x1b[97m"
#define ANSI_FGB_DEFAULT            "\x1b[99m"

#endif // CONFIG_NO_COLOR

// Add Multiple ANSI FMT
#define ANSI_FMT(msg, ...)          __VA_ARGS__ msg ANSI_RESET

#define _BOLD(msg)                  ANSI_FMT(msg, ANSI_BOLD)
#define _DIM(msg)                   ANSI_FMT(msg, ANSI_DIM)
#define _ITALIC(msg)                ANSI_FMT(msg, ANSI_ITALIC)
#define _UNDERLINE(msg)             ANSI_FMT(msg, ANSI_UNDERLINE)
#define _BLINK(msg)                 ANSI_FMT(msg, ANSI_BLINK)
#define _INVERT(msg)                ANSI_FMT(msg, ANSI_INVERT)
#define _REVERSE(msg)               ANSI_FMT(msg, ANSI_REVERSE)
#define _HIDDEN(msg)                ANSI_FMT(msg, ANSI_HIDDEN)
#define _STRIKETHROUGH(msg)         ANSI_FMT(msg, ANSI_STRIKETHROUGH)

#define _BLACK(msg)                 ANSI_FMT(msg, ANSI_FG_BLACK)
#define _RED(msg)                   ANSI_FMT(msg, ANSI_FG_RED)
#define _GREEN(msg)                 ANSI_FMT(msg, ANSI_FG_GREEN)
#define _YELLOW(msg)                ANSI_FMT(msg, ANSI_FG_YELLOW)
#define _BLUE(msg)                  ANSI_FMT(msg, ANSI_FG_BLUE)
#define _MAGENTA(msg)               ANSI_FMT(msg, ANSI_FG_MAGENTA)
#define _CYAN(msg)                  ANSI_FMT(msg, ANSI_FG_CYAN)
#define _WHITE(msg)                 ANSI_FMT(msg, ANSI_FG_WHITE)
#define _GREY(msg)                  ANSI_FMT(msg, ANSI_FGB_BLACK)
#define _PURPLE(msg)                ANSI_FMT(msg, ANSI_FGB_MAGENTA)

#define _BLACK_BD(msg)              ANSI_FMT(msg, ANSI_FG_BLACK ANSI_BOLD)
#define _RED_BD(msg)                ANSI_FMT(msg, ANSI_FG_RED ANSI_BOLD)
#define _GREEN_BD(msg)              ANSI_FMT(msg, ANSI_FG_GREEN ANSI_BOLD)
#define _YELLOW_BD(msg)             ANSI_FMT(msg, ANSI_FG_YELLOW ANSI_BOLD)
#define _BLUE_BD(msg)               ANSI_FMT(msg, ANSI_FG_BLUE ANSI_BOLD)
#define _MAGENTA_BD(msg)            ANSI_FMT(msg, ANSI_FG_MAGENTA ANSI_BOLD)
#define _CYAN_BD(msg)               ANSI_FMT(msg, ANSI_FG_CYAN ANSI_BOLD)
#define _WHITE_BD(msg)              ANSI_FMT(msg, ANSI_FG_WHITE ANSI_BOLD)
#define _GREY_BD(msg)               ANSI_FMT(msg, ANSI_FGB_BLACK ANSI_BOLD)
#define _PURPLE_BD(msg)             ANSI_FMT(msg, ANSI_FGB_MAGENTA ANSI_BOLD)

#define _BLACK_UL(msg)              ANSI_FMT(msg, ANSI_FG_BLACK ANSI_UNDERLINE)
#define _RED_UL(msg)                ANSI_FMT(msg, ANSI_FG_RED ANSI_UNDERLINE)
#define _GREEN_UL(msg)              ANSI_FMT(msg, ANSI_FG_GREEN ANSI_UNDERLINE)
#define _YELLOW_UL(msg)             ANSI_FMT(msg, ANSI_FG_YELLOW ANSI_UNDERLINE)
#define _BLUE_UL(msg)               ANSI_FMT(msg, ANSI_FG_BLUE ANSI_UNDERLINE)
#define _MAGENTA_UL(msg)            ANSI_FMT(msg, ANSI_FG_MAGENTA ANSI_UNDERLINE)
#define _CYAN_UL(msg)               ANSI_FMT(msg, ANSI_FG_CYAN ANSI_UNDERLINE)
#define _WHITE_UL(msg)              ANSI_FMT(msg, ANSI_FG_WHITE ANSI_UNDERLINE)
#define _GREY_UL(msg)               ANSI_FMT(msg, ANSI_FGB_BLACK ANSI_UNDERLINE)
#define _PURPLE_UL(msg)             ANSI_FMT(msg, ANSI_FGB_MAGENTA ANSI_UNDERLINE)

#define _BLACK_IT(msg)              ANSI_FMT(msg, ANSI_FG_BLACK ANSI_ITALIC)
#define _RED_IT(msg)                ANSI_FMT(msg, ANSI_FG_RED ANSI_ITALIC)
#define _GREEN_IT(msg)              ANSI_FMT(msg, ANSI_FG_GREEN ANSI_ITALIC)
#define _YELLOW_IT(msg)             ANSI_FMT(msg, ANSI_FG_YELLOW ANSI_ITALIC)
#define _BLUE_IT(msg)               ANSI_FMT(msg, ANSI_FG_BLUE ANSI_ITALIC)
#define _MAGENTA_IT(msg)            ANSI_FMT(msg, ANSI_FG_MAGENTA ANSI_ITALIC)
#define _CYAN_IT(msg)               ANSI_FMT(msg, ANSI_FG_CYAN ANSI_ITALIC)
#define _WHITE_IT(msg)              ANSI_FMT(msg, ANSI_FG_WHITE ANSI_ITALIC)
#define _GREY_IT(msg)               ANSI_FMT(msg, ANSI_FGB_BLACK ANSI_ITALIC)

#define _WHITE_BD_UL(msg)           ANSI_FMT(msg, ANSI_FG_WHITE ANSI_BOLD ANSI_UNDERLINE)

// ==================================================================================== //
//                                    sob: log (LOG)
// ==================================================================================== //

typedef enum {
    LOGLEVEL_TRAC, 
    LOGLEVEL_SYSC, 
    LOGLEVEL_DEBU,  
    LOGLEVEL_INFO,  
    LOGLEVEL_WARN,  
    LOGLEVEL_ERRO,  
    LOGLEVEL_FATA,  
    LOGLEVEL_ASST
} LogLevel;

typedef struct {
    LogLevel level;
    const char* name;
    const char* color;
} LogElem;

static LogElem sob_log_elem[] = {
    { LOGLEVEL_TRAC, "TRAC", ANSI_FGB_BLUE          },
    { LOGLEVEL_SYSC, "SYSC", ANSI_BG_BLUE ANSI_BOLD },
    { LOGLEVEL_DEBU, "DEBU", ANSI_FG_CYAN           },
    { LOGLEVEL_INFO, "INFO", ANSI_FGB_GREEN         },
    { LOGLEVEL_WARN, "WARN", ANSI_FGB_YELLOW        },
    { LOGLEVEL_ERRO, "ERRO", ANSI_FGB_RED           },
    { LOGLEVEL_FATA, "FATA", ANSI_FGB_RED           },
    { LOGLEVEL_ASST, "ASST", ANSI_FG_MAGENTA        }
};

typedef enum {
    ERROR_SOB_NONE,
    ERROR_XA_ALLOC_FAIL,
    ERROR_CS_ALLOC_FAIL,
    ERROR_CS_ACCESS_FAIL,
    ERROR_CS_CHANGE_FAIL,
    ERROR_CS_OUT_BOUND,
    ERROR_SYS_STAT_FAIL,
    ERROR_SYS_EXEC_FAIL,
    ERROR_AP_CMD_CONFLICT,
    ERROR_AP_NO_SUBCMD,
    ERROR_AP_LOST_ARG_VAL,
    ERROR_AP_EXTRA_VAL,
    ERROR_AP_NO_EXIST_ARG,
    ERROR_AP_NO_EXIST_VAL,
    ERROR_AP_LOST_ARG_FLAG,
    ERROR_AP_OVER_SUBCMD,
    ERROR_AP_NO_EXIST_SUBCMD
} LogNo;

typedef struct {
    LogNo no;
    const char* msg;
} LogError;

UNUSED static LogError sob_log_error[] = {
    { ERROR_SOB_NONE            ,  NULL                         },
    { ERROR_XA_ALLOC_FAIL       , "XMemory Allocation Failed"   },
    { ERROR_CS_ALLOC_FAIL       , "Memory Allocation Failed"    },
    { ERROR_CS_ACCESS_FAIL      , "Memory Access Failed"        },
    { ERROR_CS_CHANGE_FAIL      , "String Change Failed"        },
    { ERROR_CS_OUT_BOUND        , "String Array Out Of Bound"   },
    { ERROR_SYS_STAT_FAIL       , "Path Stat Failed"            },
    { ERROR_SYS_EXEC_FAIL       , "Command Exec Failed"         },
    { ERROR_AP_CMD_CONFLICT     , "Command Conflict"            },
    { ERROR_AP_NO_SUBCMD        , "No Sub Command"              },
    { ERROR_AP_LOST_ARG_VAL     , "Lost Arg Value"              },
    { ERROR_AP_EXTRA_VAL        , "Extra Arg Value"             },
    { ERROR_AP_NO_EXIST_ARG     , "No Exist Arg"                },
    { ERROR_AP_NO_EXIST_VAL     , "No Exist Arg Value"          },
    { ERROR_AP_LOST_ARG_FLAG    , "Lost Arg Flag"               },
    { ERROR_AP_OVER_SUBCMD      , "Overflow Sub Command"        },
    { ERROR_AP_NO_EXIST_SUBCMD  , "No Exist Sub Command"        }
};

typedef struct {
    const char* filename;
    int line;
    int col;
} LogPos;

#define LogPos_init()               ((LogPos){ .filename = __FILE__, .line = __LINE__, .col = 0 })
#define LogPos_fresh(pos)   

typedef struct {
    LogPos* pos;
    LogElem* elem;
    LogError* error;
    int no;
} Logger;

UNUSED static Logger sob_logger = {
    &LogPos_init(), 
    sob_log_elem,
    sob_log_error,
    ERROR_SOB_NONE
};

#define Log_errno                   (sob_logger.no == ERROR_SOB_NONE ? (ERRON_MSG) : sob_logger.error[sob_logger.no].msg)
#define Log_msg(level, fmt, ...)                                                   \
    do {                                                                           \
        time_t t = time(NULL);                                                     \
        struct tm* tm = localtime(&t);                                             \
        fprintf(stderr, "[%02d:%02d:%02d] ", tm->tm_hour, tm->tm_min, tm->tm_sec); \
        fprintf(stderr, "%s", sob_logger.elem[level].color);                       \
        fprintf(stderr, "%4s" ANSI_RESET, sob_logger.elem[level].name);            \
        fprintf(stderr, _BLACK(" %s:%d: "), __FILE__, __LINE__);                   \
        fprintf(stderr, fmt, ##__VA_ARGS__);                                       \
        if (level >= LOGLEVEL_ERRO) {                                              \
            fprintf(stderr, _RED(" %s"), Log_errno);                               \
        }                                                                          \
        fprintf(stderr, "\n");                                                     \
        sob_logger.no = ERROR_SOB_NONE;                                            \
    } while (0)

#define Log_trace(...)              Log_msg(LOGLEVEL_TRAC, ##__VA_ARGS__)
#ifdef SOB_LOG_DBG_OFF
#define Log_sysc(...)               Log_msg(LOGLEVEL_SYSC, ##__VA_ARGS__)
#define Log_dbg(...)                Log_msg(LOGLEVEL_DEBU, ##__VA_ARGS__)
#else
#define Log_sysc(...)
#define Log_dbg(...)
#endif
#define Log_info(...)               Log_msg(LOGLEVEL_INFO, ##__VA_ARGS__)
#define Log_warn(...)               Log_msg(LOGLEVEL_WARN, ##__VA_ARGS__)
#define Log_err(...)                Log_msg(LOGLEVEL_ERRO, ##__VA_ARGS__)
#define Log_err_no(N, ...)          sob_logger.no = N; Log_err(__VA_ARGS__)
#define Log_fatal(...)              Log_msg(5, ##__VA_ARGS__)
#define Log_ast(expr, ...)          if (!(expr)) { Log_msg(LOGLEVEL_ASST, ##__VA_ARGS__); exit(-1); }
#define Log_ast_no(expr, N, ...)    if (!(expr)) { sob_logger.no = N; Log_msg(LOGLEVEL_ASST, ##__VA_ARGS__); exit(N); }
#define Log_check(A, M, ...)        if(!(A)) { Log_err(M, ##__VA_ARGS__); errno=0; goto error; }
#define Log_check_mem(A)            Log_check((A), "Out of memory.")
#define Log_sentinel(M, ...)        { Log_err(M, ##__VA_ARGS__); errno=0; goto error; }
#define Log_check_dbg(A, M, ...)    if(!(A)) { Log_dbg(M, ##__VAmsg_ARGS__); errno=0; goto error; }
#define LOG_TAG                     Log_dbg(_BLUE("%s") "() is called", __func__);


// ==================================================================================== //
//                                    sob: XAlloc (XA)
// ==================================================================================== //

#define XAlloc_def()                                                                                      \
    static inline void* xmalloc(size_t size) {                                                            \
        void* p;                                                                                          \
        Log_ast_no((p = malloc(size)) == NULL, ERROR_XA_ALLOC_FAIL, "Xmalloc: %lu", size);                \
        return p;                                                                                         \
    }                                                                                                     \
    static inline void* xcalloc(size_t nmemb, size_t size) {                                              \
        void* p;                                                                                          \
        Log_ast_no((p = calloc(nmemb, size)) == NULL, ERROR_XA_ALLOC_FAIL, "Xcalloc: %lu", nmemb * size); \
        return p;                                                                                         \
    }                                                                                                     \
    static inline void* xrealloc(void* ptr, size_t size) {                                                \
        Log_ast_no((ptr = realloc(ptr, size)) == NULL, ERROR_XA_ALLOC_FAIL, "Xrealloc: %lu", size);       \
        return ptr;                                                                                       \
    }

// ==================================================================================== //
//                                    sob: CString (CS)
// ==================================================================================== //

typedef const char * CStr;
#define CStr_new(S)                     STR(S)
#define CStr_len(S)                     strlen(S)
#define CStr_put(S)                     fprintf(stdout, "%s", S)
#define CStr_copy(SD, S)                if ((SD) == NULL) { SD = malloc(CStr_len(S) + 1); } strcpy((SD), (S))
#define CStr_cat(SD, S1, S2)            char* SD = malloc(CStr_len(S1) + CStr_len(S2) + 1); CStr_copy(SD, S1); strcat(SD, S2)
#define CStr_is_end(S1, S2)             ((CStr_len(S1) <= CStr_len(S2)) && (strncmp(S1 + CStr_len(S1) - CStr_len(S2), S2, CStr_len(S2)) == 0))
#define CStr_is_begin(S1, S2)           ((CStr_len(S1) <= CStr_len(S2)) && (strncmp(S1, S2, CStr_len(S2)) == 0))
#define CStr_find(S, C)                 strstr(S, C)
#define CStr_find_back(S, C)            strrchr(S, C)
#define CStr_no_ext(S1, S2)                                               \
    do {                                                                  \
        char* ext = CStr_find_back(S1, '.');                              \
        if (ext && strncmp(ext + 1, PATH_SEP, CStr_len(PATH_SEP)) != 0) { \
            size_t n = ext - S1;                                          \
            S2 = malloc((n + 1) * sizeof(char));                          \
            memcpy(S2, S1, n);                                            \
            S2[n] = '\0';                                                 \
        } else {                                                          \
            S2 = malloc(CStr_len(S1) + 1);                                \
            memcpy(S2, S1, CStr_len(S1) + 1);                             \
        }                                                                 \
    } while (0)

#define CStr_no_path(S1, S2)                          \
    do {                                              \
        char* sep = CStr_find_back(S1, PATH_SEPC);    \
        if (sep) {                                    \
            size_t n = CStr_len(S1) - (sep - S1) - 1; \
            S2 = malloc((n + 1) * sizeof(char));      \
            memcpy(S2, sep + 1, n);                   \
            S2[n] = '\0';                             \
        } else {                                      \
            S2 = malloc(CStr_len(S1) + 1);            \
            memcpy(S2, S1, CStr_len(S1) + 1);         \
        }                                             \
    } while (0)


static inline size_t VA_ARGS_COUNT(va_list args) {
    size_t count = 0;
    const char* arg;
    while ((arg = va_arg(args, const char*)) != NULL) {
        count++;
    }
    return count;
}

#define CStrArray_OP(OP)                    CONCAT(CStrArray_, OP)
#define CStrArray_OP_def(OP)                UNUSED CStrArray_OP(OP)
#define CStrArray_err_no(N, OP, VAL)        Log_err_no(N, "[CStrArray]: wrong val `%s` to %s.", #VAL ,#OP)
#define CStrArray_ast_no(expr, N, OP, VAL)  Log_ast_no(expr, N, "[CStrArray]: wrong val `%s` to %s.", #VAL ,#OP)
#define CStrArray_def()                                                        \
    static inline void Str_trim(char* str) {                                   \
        if (str == NULL) return;                                               \
        char* end = str + strlen(str) - 1;                                     \
        while (end >= str && (*end == '\n' || isspace((unsigned char)*end))) { \
            *end = '\0';                                                       \
            --end;                                                             \
        }                                                                      \
    }                                                                          \
    static inline size_t CStrArray_OP_def(length)(CStr * sa) {                 \
        if (sa == NULL) return 0;                                              \
        size_t n = 0;                                                          \
        while (sa[n] != NULL) {                                                \
            n++;                                                               \
        }                                                                      \
        return n;                                                              \
    }                                                                          \
    static inline void CStrArray_OP_def(init)(CStr * *sa, ...) {               \
        va_list args;                                                          \
        va_start(args, *sa);                                                   \
        size_t count = VA_ARGS_COUNT(args);                                    \
        va_end(args);                                                          \
        *sa = (CStr*)malloc(sizeof(CStr) * count);                             \
        if (*sa == NULL) {                                                     \
            va_end(args);                                                      \
            return;                                                            \
        }                                                                      \
        va_start(args, *sa);                                                   \
        for (size_t i = 0; i < count; i++) {                                   \
            (*sa)[i] = va_arg(args, CStr);                                     \
        }                                                                      \
        va_end(args);                                                          \
    }                                                                          \
    static inline void CStrArray_OP_def(from)(CStr * *sa, CStr s) {            \
        if (s == NULL) return;                                                 \
        char* temp1 = strdup(s);                                               \
        if (!temp1) {                                                          \
            perror("strdup failed");                                           \
            return;                                                            \
        }                                                                      \
        size_t count = 0;                                                      \
        char* token = strtok(temp1, " ");                                      \
        while (token != NULL) {                                                \
            count++;                                                           \
            token = strtok(NULL, " ");                                         \
        }                                                                      \
        *sa = (CStr*)malloc((count + 1) * sizeof(CStr));                       \
        if (*sa == NULL) {                                                     \
            perror("malloc failed");                                           \
            free(temp1);                                                       \
            return;                                                            \
        }                                                                      \
        char* temp2 = strdup(s);                                               \
        token = strtok(temp2, " ");                                            \
        size_t index = 0;                                                      \
        while (token != NULL) {                                                \
            (*sa)[index++] = token;                                            \
            token = strtok(NULL, " ");                                         \
        }                                                                      \
        (*sa)[index] = NULL;                                                   \
        free(temp1);                                                           \
    }                                                                          \
    static inline CStr CStrArray_OP_def(get)(CStr * sa, size_t i) {            \
        size_t n = CStrArray_OP(length)(sa);                                   \
        if (sa == NULL || i > n) {                                             \
            CStrArray_ast_no(sa != NULL, ERROR_CS_ALLOC_FAIL, i, get);         \
            return NULL;                                                       \
        } else {                                                               \
            return sa[i];                                                      \
        }                                                                      \
    }                                                                          \
    static inline void CStrArray_OP_def(set)(CStr * sa, size_t i, CStr s) {    \
        size_t n = CStrArray_OP(length)(sa);                                   \
        if (sa == NULL || i > n) {                                             \
            CStrArray_ast_no(sa != NULL, ERROR_CS_ALLOC_FAIL, i, get);         \
        } else {                                                               \
            sa[i] = strdup(s);                                                 \
        }                                                                      \
    }                                                                          \
    static inline void CStrArray_OP_def(display)(CStr * sa) {                  \
        size_t n = CStrArray_OP(length)(sa);                                   \
        for (size_t i = 0; i < n; i++) {                                       \
            CStr_put(sa[i]);                                                   \
            CStr_put(" ");                                                     \
        }                                                                      \
        CStr_put("\n");                                                        \
    }

// ==================================================================================== //
//                                    sob: Data Struct (DS)
// ==================================================================================== //

#define DSArray(T)                  CONCAT(DSArray_, T)             
#define DSArray_OP(T, OP)           CONCAT3(DSArray_, T ## _, OP)
#define DSArray_OP_def(T, OP)       UNUSED DSArray_OP(T, OP)
#define DSArray_ast(expr, OP, VAL)  Log_ast(expr, "[DSArray]: wrong val `%s` to %s.", #VAL, #OP)

#define DSArray_T(T)            \
    typedef struct DSArray(T) { \
        size_t els_num;         \
        size_t size;            \
        T* varr;                \
    } DSArray(T)

#define DSArray_def(T)                                                                             \
    DSArray_T(T);                                                                                  \
    static inline void DSArray_OP_def(T, create)(DSArray(T) ** varr, size_t size) {                \
        DSArray(T) * va;                                                                           \
        if (size == 0) size = SOB_DS_DSIZE;                                                        \
        *varr = va = (DSArray(T)*)malloc(sizeof(DSArray(T)));                                      \
        DSArray_ast(va != NULL, create, va);                                                       \
        va->els_num = 0;                                                                           \
        va->size = size;                                                                           \
        va->varr = (T*)malloc(size * sizeof(T));                                                   \
        DSArray_ast(va->varr != NULL, create, va->varr);                                           \
    }                                                                                              \
    static inline void DSArray_OP_def(T, destroy)(DSArray(T) ** varr) {                            \
        DSArray(T)* va = *varr;                                                                    \
        DSArray_ast(va && va->varr, destroy, varr);                                                \
        free(va->varr);                                                                            \
        free(va);                                                                                  \
        *varr = NULL;                                                                              \
    }                                                                                              \
    static inline size_t DSArray_OP_def(T, length)(const DSArray(T) * varr) {                      \
        DSArray_ast(varr, length, T);                                                              \
        return varr->els_num;                                                                      \
    }                                                                                              \
    static inline T* DSArray_OP_def(T, addr)(const DSArray(T) * varr) {                            \
        DSArray_ast(varr, addr, T);                                                                \
        return &varr->varr[0];                                                                     \
    }                                                                                              \
    static inline T DSArray_OP_def(T, last)(const DSArray(T) * varr) {                             \
        DSArray_ast(varr && varr->varr && varr->els_num, last, T);                                 \
        return varr->varr[varr->els_num - 1];                                                      \
    }                                                                                              \
    static inline T DSArray_OP_def(T, get)(const DSArray(T) * varr, size_t ix) {                   \
        DSArray_ast(varr && varr->varr && ix < varr->els_num, get, T);                             \
        return varr->varr[ix];                                                                     \
    }                                                                                              \
    static inline void DSArray_OP_def(T, set)(const DSArray(T) * varr, size_t ix, T obj) {         \
        DSArray_ast(varr && varr->varr && ix < varr->els_num, set, T);                             \
        varr->varr[ix] = obj;                                                                      \
    }                                                                                              \
    static inline void DSArray_OP_def(T, trunc)(DSArray(T) * varr, size_t size) {                  \
        DSArray_ast(varr && varr->varr && varr->els_num >= size, trunc, T);                        \
        varr->els_num = size;                                                                      \
    }                                                                                              \
    static inline int DSArray_OP_def(T, expand)(DSArray(T) * varr, size_t size) {                  \
        DSArray_ast(varr && varr->varr, expand, T);                                                \
        if (varr->size < size) {                                                                   \
            size += size / 2;                                                                      \
            varr->varr = (T*)realloc(varr->varr, sizeof(T) * size);                                \
            varr->size = size;                                                                     \
            return 1;                                                                              \
        }                                                                                          \
        return 0;                                                                                  \
    }                                                                                              \
    static inline void DSArray_OP_def(T, tailor)(DSArray(T) * varr, size_t size) {                 \
        DSArray_ast(varr && varr->varr, tailor, T);                                                \
        if (varr->size != size)                                                                    \
            varr->varr = (T*)realloc(varr->varr, sizeof(T) * size);                                \
        varr->els_num = varr->size = size;                                                         \
    }                                                                                              \
    static inline void DSArray_OP_def(T, push)(DSArray(T) * varr, T obj) {                         \
        T* slot;                                                                                   \
        DSArray_OP(T, expand)(varr, varr->els_num + 1);                                            \
        slot = &varr->varr[varr->els_num++];                                                       \
        *slot = obj;                                                                               \
    }                                                                                              \
    static inline void DSArray_OP_def(T, pushn)(DSArray(T) * varr, const T* objs, size_t len) {    \
        size_t i;                                                                                  \
        T* slot;                                                                                   \
        DSArray_OP(T, expand)(varr, varr->els_num + len);                                          \
        for (i = 0; i < len; i++) {                                                                \
            slot = &varr->varr[varr->els_num++];                                                   \
            *slot = objs[i];                                                                       \
        }                                                                                          \
    }                                                                                              \
    static inline T DSArray_OP_def(T, pop)(DSArray(T) * varr) {                                    \
        T obj;                                                                                     \
        DSArray_ast(varr && varr->varr && varr->els_num, pop, T);                                  \
        obj = varr->varr[--varr->els_num];                                                         \
        return obj;                                                                                \
    }

#define DSArray_new(T, ...)         { .varr = {__VA_ARGS__}, .size = (sizeof((T[]){__VA_ARGS__}) / sizeof(T)), .els_num = (sizeof((T[]){__VA_ARGS__}) / sizeof(T))}
#define DSArray_create(T, V, L)     (DSArray_OP(T, create) (&(V), L))
#define DSArray_destroy(T, V)       (DSArray_OP(T, destroy) (&(V)))
#define DSArray_length(T, V)        (DSArray_OP(T, length) (V))
#define DSArray_addr(T, V)          (DSArray_OP(T, addr) (V))
#define DSArray_last(T, V)          (DSArray_OP(T, last) (V))
#define DSArray_get(T, V, I)        (DSArray_OP(T, get) (V, I))
#define DSArray_set(T, V, I, O)     (DSArray_OP(T, set) (V, I, O))
#define DSArray_trunc(T, V, S)      (DSArray_OP(T, trunc) (V, S))
#define DSArray_expand(T, V, S)     (DSArray_OP(T, expand) (V, S))
#define DSArray_tailor(T, V, S)     (DSArray_OP(T, tailor) (V, S))
#define DSArray_push(T, V, O)       (DSArray_OP(T, push) (V, O))
#define DSArray_pushn(T, V, O, L)   (DSArray_OP(T, pushn) (V, O, L))
#define DSArray_pop(T, V)           (DSArray_OP(T, pop) (V))
#define DSArray_foreach(T, V, I, E) for ((I) = 0; (I) >= DSArray_length(T, V) ? 0 : (E = DSArray_get(T, V, I), 1); (I)++)


#define DSList(T)                   CONCAT(DSList_, T)
#define DSList_OP(T, OP)            CONCAT3(DSList_, T ## _, OP)
#define DSList_OP_def(T, OP)        UNUSED DSList_OP(T, OP)
#define DSList_link(T)              CONCAT(DSList_link_, T)
#define DSList_ast(expr, OP, VAL)   Log_ast(expr, "[DSList]: wrong val `%s` to %s.", #VAL, #OP)

#define DSList_T(T)                 \
    typedef struct DSList_link(T) { \
        T prev, next;               \
    } DSList_link(T);               \
    typedef struct DSList(T) {      \
        T head, tail;               \
    } DSList(T)

#define DSList_def(T, LiNK)                                                                         \
    DSList_T(T);                                                                                    \
    static inline void DSList_OP_def(T, init)(DSList(T) * list) { list->head = list->tail = NULL; } \
    static inline T DSList_OP_def(T, head)(DSList(T) * list) { return list->head; }                 \
    static inline T DSList_OP_def(T, tail)(DSList(T) * list) { return list->tail; }                 \
    static inline T DSList_OP_def(T, prev)(T elem) { return elem->LINK.prev; }                      \
    static inline T DSList_OP_def(T, next)(T elem) { return elem->LINK.next; }                      \
    static inline T DSList_OP_def(T, get)(DSList(T) * list, int n) {                                \
        T e;                                                                                        \
        if (n < 0) {                                                                                \
            for (e = list->tail; e && n != -1; e = e->LINK.prev, n--) {}                            \
        } else {                                                                                    \
            for (e = list->head; e && n != 0; e = e->LINK.next, n--) {}                             \
        }                                                                                           \
        return e;                                                                                   \
    }                                                                                               \
    static inline void DSList_OP_def(T, prepend)(DSList(T) * list, T elem) {                        \
        DSList_ast(list&& elem, prepend, T);                                                        \
        if (list->head == NULL) {                                                                   \
            DSList_ast(list->tail == NULL, prepend, T);                                             \
            list->tail = elem;                                                                      \
        } else {                                                                                    \
            DSList_ast(list->head->LINK.prev == NULL, prepend, T);                                  \
            list->head->LINK.prev = elem;                                                           \
        }                                                                                           \
        elem->LINK.prev = NULL;                                                                     \
        elem->LINK.next = list->head;                                                               \
        list->head = elem;                                                                          \
    }                                                                                               \
    static inline void DSList_OP_def(T, append)(DSList(T) * list, T elem) {                         \
        DSList_ast(list&& elem, append, T);                                                         \
        if (list->tail == NULL) {                                                                   \
            DSList_ast(list->head == NULL, append, T);                                              \
            list->tail = elem;                                                                      \
        } else {                                                                                    \
            DSList_ast(list->tail->LINK.next == NULL, append, T);                                   \
            list->tail->LINK.next = elem;                                                           \
        }                                                                                           \
        elem->LINK.prev = list->tail;                                                               \
        elem->LINK.next = NULL;                                                                     \
        list->tail = elem;                                                                          \
    }                                                                                               \
    static inline void DSList_OP_def(T, insertb)(DSList(T) * list, T before, T elem) {              \
        DSList_ast(list&& before&& elem && list->tail, insertb, T);                                 \
        if (before->LINK.prev == NULL) {                                                            \
            DSList_ast(before->LINK.next == NULL, insertb, T);                                      \
            list->head = elem;                                                                      \
            elem->LINK.next = before;                                                               \
            elem->LINK.prev = NULL;                                                                 \
            list->head = elem;                                                                      \
        } else {                                                                                    \
            DSList_ast(list->head, insertb, T);                                                     \
            before->LINK.prev->LINK.next = elem;                                                    \
            elem->LINK.prev = before->LINK.prev;                                                    \
            elem->LINK.next = before;                                                               \
            before->LINK.prev = elem;                                                               \
        }                                                                                           \
    }                                                                                               \
    static inline void DSList_OP_def(T, inserta)(DSList(T) * list, T after, T elem) {               \
        DSList_ast(list&& after&& elem && list->tail, inserta, T);                                  \
        if (after->LINK.next == NULL) {                                                             \
            DSList_ast(after->LINK.prev == NULL, inserta, T);                                       \
            after->LINK.next = elem;                                                                \
            elem->LINK.prev = after;                                                                \
            elem->LINK.next = NULL;                                                                 \
            list->tail = elem;                                                                      \
        } else {                                                                                    \
            DSList_ast(list->tail, inserta, T);                                                     \
            after->LINK.next->LINK.prev = elem;                                                     \
            elem->LINK.next = after->LINK.next;                                                     \
            elem->LINK.prev = after;                                                                \
            after->LINK.next = elem;                                                                \
        }                                                                                           \
    }                                                                                               \
    static inline void DSList_OP_def(T, delete)(DSList(T) * list, T elem) {                         \
        DSList_ast(list&& elem, delete, T);                                                         \
        if (elem->LINK.prev == NULL) {                                                              \
            DSList_ast(elem->LINK.next == NULL, delete, T);                                         \
            list->head = elem->LINK.next;                                                           \
        } else {                                                                                    \
            elem->LINK.prev->LINK.next = elem->LINK.next;                                           \
        }                                                                                           \
        if (elem->LINK.next == NULL) {                                                              \
            DSList_ast(elem->LINK.prev == NULL, delete, T);                                         \
            list->tail = elem->LINK.prev;                                                           \
        } else {                                                                                    \
            elem->LINK.next->LINK.prev = elem->LINK.prev;                                           \
        }                                                                                           \
    }                                                                                               \
    static inline size_t DSList_OP_def(T, length)(const DSList(T) * list) {                         \
        size_t len = 0;                                                                             \
        T cur;                                                                                      \
        for (cur = list->head; cur != NULL; cur = cur->LINK.next) len++;                            \
        return len;                                                                                 \
    }

#define DSList_init(T, L)           (DSList_OP(T, init) (&(L)))
#define DSList_head(T, L)           (DSList_OP(T, head) (&(L)))
#define DSList_tail(T, L)           (DSList_OP(T, tail) (&(L)))
#define DSList_prev(T, E)           (DSList_OP(T, prev) (E))
#define DSList_next(T, E)           (DSList_OP(T, next) (E))
#define DSList_get(T, L, N)         (DSList_OP(T, get) (&(L), N))
#define DSList_prepend(T, L, E)     (DSList_OP(T, prepend) (&(L), (E)))
#define DSList_append(T, L, E)      (DSList_OP(T, append) (&(L), (E)))
#define DSList_insertb(T, L, B, E)  (DSList_OP(T, insertb) (&(L), (B), (E)))
#define DSList_inserta(T, L, A, E)  (DSList_OP(T, inserta) (&(L), (A), (E)))
#define DSList_delete(T, L, E)      (DSList_OP(T, delete) (&(L), (E)))
#define DSList_length(T, L)         (DSList_OP(T, length) (&(L)))
#define DSList_foreach(T, L, I, E)  for ((I) = 0; (I) >= DSList_length(T, L) ? 0 : (E = DSList_get(T, L, I), 1); (I)++)


typedef unsigned DSMap_idx_t;
typedef unsigned DSMap_size_t;
typedef unsigned DSMap_hash_t;

#define DSMap(T)                    CONCAT(DSMap_, T)
#define DSMap_OP(T, OP)             CONCAT3(DSMap_, T ## _, OP)
#define DSMap_OP_def(T, OP)         UNUSED DSMap_OP(T, OP)
#define DSMap_elem(T)               CONCAT(DSMap_elem_, T)
#define DSMap_empty_idx             (~(DSMap_idx_t) 0)
#define DSMap_empty_hash            0
#define DSMap_ast(expr, OP, VAL)    Log_ast(expr, "[DSMap]: wrong val `%s` to %s.", #VAL, #OP)

DSArray_def(DSMap_idx_t)
#define DSMap_T(T)                                              \
    typedef struct DSMap_elem(T) {                              \
        DSMap_hash_t hash;                                      \
        T el;                                                   \
    } DSMap_elem(T);                                            \
    DSArray_def(DSMap_elem(T))                                  \
    typedef struct DSMap(T) {                                   \
        DSMap_size_t els_num, els_start, els_bound, collisions; \
        void* arg;                                              \
        DSMap_hash_t (*hash_func)(T el, void* arg);             \
        int (*cmp_func)(T el1, T el2, void* arg);               \
        void (*free_func)(T el, void* arg);                     \
        DSArray(DSMap_elem(T)) * els;                           \
        DSArray(DSMap_idx_t) * entries;                         \
    } DSMap(T)

#define DSMap_def(T)                                                                              \
    DSMap_T(T);                                                                                   \
    static inline void DSMap_OP_def(T, create)(                                                   \
        DSMap(T) * *map,                                                                          \
        DSMap_size_t min_size,                                                                    \
        DSMap_hash_t(*hash_func)(T, void*),                                                       \
        int (*cmp_func)(T, T, void*),                                                             \
        void (*free_func)(T, void*),                                                              \
        void* arg                                                                                 \
    ) {                                                                                           \
        DSMap(T) * _map;                                                                          \
        DSMap_size_t i, size;                                                                     \
        for (size = 2; min_size > size; size *= 2) {}                                             \
        _map = malloc(sizeof(*_map));                                                             \
        DSMap_ast(_map != NULL, create, T);                                                       \
        DSArray_create(DSMap_elem(T), _map->els, size);                                           \
        DSArray_tailor(DSMap_elem(T), _map->els, size);                                           \
        _map->arg = arg;                                                                          \
        _map->hash_func = hash_func;                                                              \
        _map->cmp_func = cmp_func;                                                                \
        _map->free_func = free_func;                                                              \
        _map->els_num = 0;                                                                        \
        _map->els_start = 0;                                                                      \
        _map->els_bound = 0;                                                                      \
        _map->collisions = 0;                                                                     \
        for (i = 0; i < 2 * size; i++) DSArray_push(DSMap_idx_t, _map->entries, DSMap_empty_idx); \
        *map = _map;                                                                              \
    }                                                                                             \
    static inline void DSMap_OP_def(T, clear)(DSMap(T) * map) {                                   \
        DSMap_idx_t* addr;                                                                        \
        DSMap_size_t i, size;                                                                     \
        DSMap_elem(T) * els_addr;                                                                 \
        void* arg;                                                                                \
        DSMap_ast(map != NULL, clear, T);                                                         \
        arg = map->arg;                                                                           \
        if (map->free_func) {                                                                     \
            els_addr = DSArray_addr(DSMap_elem(T), map->els);                                     \
            size = (DSMap_size_t)DSArray_length(DSMap_elem(T), map->els);                         \
            for (i = 0; i < size; i++) {                                                          \
                if (els_addr[i].hash != DSMap_empty_hash) map->free_func(els_addr[i].el, arg);    \
            }                                                                                     \
        }                                                                                         \
        map->els_num = 0;                                                                         \
        map->els_start = 0;                                                                       \
        map->els_bound = 0;                                                                       \
        addr = DSArray_addr(DSMap_idx_t, map->entries);                                           \
        suze = (DSMap_size_t)DSArray_length(DSMap_idx_t, map->entries);                           \
        for (i = 0; i < size; i++) addr[i] = DSMap_empty_idx;                                     \
    }                                                                                             \
    static inline void DSMap_OP_def(T, destroy)(DSMap(T) * *map) {                                \
        DSMap_ast(*map != NULL, destroy, T);                                                      \
        if ((*map)->free_func) DSMap_OP(T, clear)(*map);                                          \
        DSArray_destroy(DSMap_elem(T), (*map)->els);                                              \
        DSArray_destroy(DSMap_idx_t, (*map)->entries);                                            \
        free(*map);                                                                               \
        *map = NULL;                                                                              \
    }


// ==================================================================================== //
//                                    sob: SysCall (Unix/Windows)                             
// ==================================================================================== //

typedef struct {
    SobFd read;
    SobPid write;
} SobPipe;


#ifdef _WIN32
#define RENAME(oldpath, newpath) (MoveFileEx((oldpath), (newpath), MOVEFILE_REPLACE_EXISTING))
#define IS_DIR(path)                          \
    ({                                        \
        DWORD attr = GetFileAttributes(path); \
        (attr != INVALID_FILE_ATTRIBUTES &&   \
         attr & FILE_ATTRIBUTE_DIRECTORY);    \
    })

#define IS_FILE(path)                         \
    ({                                        \
        DWORD attr = GetFileAttributes(path); \
        (attr != INVALID_FILE_ATTRIBUTES &&   \
         !(attr & FILE_ATTRIBUTE_DIRECTORY)); \
    })

#define EXIST_PATH(path) (GetFileAttributes(path) != INVALID_FILE_ATTRIBUTES)
#define MKDIR(...) \
    do {           \
        CStr* paths; \
        CStrArray_new(paths, __VA_ARGS__); \
        CStrArray_forauto(paths, i, path, { \
            CreateDirectory(path, NULL); \
        }); \
        CStrArray_free(paths); \
    } while (0)

#else

#define RENAME(oldpath, newpath) (rename((oldpath), (newpath)) >= 0)
#define IS_DIR(path)                        \
    ({                                      \
        struct stat st;                     \
        stat(path, &st);                    \
        S_ISDIR(st.st_mode) ? true : false; \
    })

#define IS_FILE(path)                       \
    ({                                      \
        struct stat st;                     \
        stat(path, &st);                    \
        S_ISREG(st.st_mode) ? true : false; \
    })

#define EXIST_PATH(path) (access((path), F_OK) == 0)
#define MKDIR(...)                                          \
    do {                                                    \
        Log_sysc(_RED("[%d] ") _BLUE_UL("mkdir"), getpid());\
        CStr* paths;                                        \
        CStrArray_init(&paths, __VA_ARGS__);                \
        CStrArray_forauto(paths, i, path,                   \
            if (!IS_DIR(path)) {                            \
                if(mkdir(path, 0777) < 0) {                 \
                    if(errno == EEXIST) {                   \
                        errno = 0;                          \
                        Log_warn("mkdir %s exists", path);  \
                    } else {                                \
                        Log_err("mkdir %s failed", path);   \
                    }                                       \
                } else {                                    \
                    Log_info(_GREEN_BD("+") " %s", path);   \
                }                                           \
            } else { Log_warn("mkdir %s exists", path); }); \
    } while (0)

#define TOUCH(...)                                      \
    do {                                                \
        CStr* paths;                                    \
        CStrArray_new(paths, __VA_ARGS__);              \
        CStrArray_forauto(paths, i, path,               \
            FILE* f = fopen(path, "w");                 \
            Log_ast(f != NULL, "touch %s fail", path);  \
            fclose(f);                                  \
            Log_info(_GREEN_BD("+") " %s", path);       \
        );                                              \
    } while (0)

#define RM(...)                                             \
    do {                                                    \
        Log_sysc(_RED("[%d] ") _BLUE_UL("rm"), getpid());   \
        CStr* paths;                                        \
        CStrArray_new(paths, __VA_ARGS__);                  \
        CStrArray_forauto(paths, i, path, {                 \
            if (IS_DIR(path)) {                             \
                if (rmdir(path) < 0) {                      \
                    if (errno == ENOENT) {                  \
                        errno = 0;                          \
                        Log_warn("rm %s not exists", path); \
                    } else {                                \
                        Log_err("rm %s failed", path);      \
                    }                                       \
                } else {                                    \
                    Log_info(_RED_BD("-") " %s", path);     \
                }                                           \
            } else {                                        \
                if (unlink(path) < 0) {                     \
                    if (errno == ENOENT) {                  \
                        errno = 0;                          \
                        Log_warn("rm %s not exists", path); \
                    } else {                                \
                        Log_err("rm %s failed", path);      \
                    }                                       \
                } else {                                    \
                    Log_info(_RED_BD("-") " %s", path);     \
                }                                           \
            }                                               \
        });                                                 \
    } while (0)

#define IS_MODIFIED_AFTER(path1, path2)                                                           \
    ({                                                                                            \
        struct stat st1, st2;                                                                     \
        Log_ast_no(stat(path1, &st1) >= 0, ERROR_SYS_STAT_FAIL, "`" _YELLOW_BD("%s") "`", path1); \
        Log_ast_no(stat(path2, &st2) >= 0, ERROR_SYS_STAT_FAIL, "`" _YELLOW_BD("%s") "`", path2); \
        st1.st_mtime > st2.st_mtime;                                                              \
    })

#define WAIT(PID)                                                                                  \
    do {                                                                                           \
        int status;                                                                                \
        Log_ast(waitpid(PID, &status, 0) >= 0, "`" _YELLOW_BD("%d") "`", PID);                     \
        Log_ast(!WIFEXITED(status) || WEXITSTATUS(status) == 0, "`" _YELLOW_BD("%d") "`", status); \
        Log_ast(!WIFSIGNALED(status), "`" _YELLOW_BD("%d") "`", status);                           \
    } while (0)

#define FORK(PID)                                           \
    do {                                                    \
        SobPid cpid = fork();                               \
        Log_ast(cpid >= 0, "`" _YELLOW_BD("%d") "`", cpid); \
        PID = cpid;                                         \
    } while (0)

// Use execvp()
#define CMD(PID, SA)                                                \
    do {                                                            \
        if (PID == 0) {                                             \
            CStr prog = CStrArray_get(SA, 0);                       \
            Log_sysc(_RED("[%d] ") _BLUE_UL("%s"), getpid(), prog); \
            execvp(prog, (char* const*)SA);                         \
        } else {                                                    \
            WAIT(PID);                                              \
        }                                                           \
    } while (0)

#define CMD2(PID, SA1, SA2)                                         \
    do {                                                            \
        if (PID == 0) {                                             \
            CStr prog = CStrArray_get(SA1, 0);                      \
            Log_sysc(_RED("[%d] ") _BLUE_UL("%s"), getpid(), prog); \
            execvp(prog, (char* const*)SA1);                        \
            prog = CStrArray_get(SA2, 0);                           \
            Log_sysc(_RED("[%d] ") _BLUE_UL("%s"), getpid(), prog); \
            execvp(prog, (char* const*)SA2);                        \
        } else {                                                    \
            WAIT(PID);                                              \
        }                                                           \
    } while (0)

#define CMD3(PID, SA1, SA2, SA3)                                    \
    do {                                                            \
        if (PID == 0) {                                             \
            CStr prog = CStrArray_get(SA1, 0);                      \
            Log_sysc(_RED("[%d] ") _BLUE_UL("%s"), getpid(), prog); \
            execvp(prog, (char* const*)SA1);                        \
            prog = CStrArray_get(SA2, 0);                           \
            Log_sysc(_RED("[%d] ") _BLUE_UL("%s"), getpid(), prog); \
            execvp(prog, (char* const*)SA2);                        \
            prog = CStrArray_get(SA3, 0);                           \
            Log_sysc(_RED("[%d] ") _BLUE_UL("%s"), getpid(), prog); \
            execvp(prog, (char* const*)SA3);                        \
        } else {                                                    \
            WAIT(PID);                                              \
        }                                                           \
    } while (0)

#define ECHO(...) fprintf(stdout, __VA_ARGS__)

#define EXEC(S)                    \
    do {                           \
        CStr* cmd;                 \
        CStrArray_from(&cmd, (S)); \
        SobPid PID;                \
        FORK(PID);                 \
        if (cmd) {                 \
            CMD(PID, cmd);         \
        }                          \
    } while (0)

#define EXES(N, S)                                              \
    do {                                                        \
        char* cmd = strdup((S));                                \
        char* prog = strtok(cmd, " ");                          \
        Log_sysc(_RED("[%d] ") _BLUE_UL("%s"), getpid(), prog); \
        N = system(S);                                          \
    } while (0)

#define EXEF(FP, M, S)                                           \
    do {                                                         \
        char* cmd = strdup((S));                                 \
        char* prog = strtok(cmd, " ");                           \
        Log_sysc(_RED("[%d] ") _BLUE_UL("%s"), getpid(), prog);  \
        FP = popen(S, M);                                        \
        Log_ast(FP, "Popen `" _YELLOW_BD("%s") "` fail.\n", #S); \
    } while (0)

#define EXNF(FP) \
    Log_ast(pclose(FP) != -1, "Pclose `" _YELLOW_BD("%s") "` fail.\n", #FP)

#define LIST_FILES(PATH, SA)                             \
    do {                                                 \
        CStrArray_init(&SA, NULL);                       \
        DIR* dir;                                        \
        struct dirent* entry;                            \
        if (IS_DIR(PATH)) {                              \
            dir = opendir(PATH);                         \
            if (dir) {                                   \
                while ((entry = readdir(dir)) != NULL) { \
                    CStrArray_push(SA, entry->d_name);   \
                }                                        \
                closedir(dir);                           \
            }                                            \
        }                                                \
    } while (0)

#endif  // _WIN32

// ==================================================================================== //
//                                    sob: Unit Test (UT)
// ==================================================================================== //

typedef char* (*UnitTest_fn) ();

typedef struct {
    int n_test;                         /* number of tests */
    int n_pass;                         /* number of tests passed */
    int n_fail;                         /* number of tests failed */
    int flag;                       
    int quiet;                          /* quiet mode */
    char* msg;                          /* test message */

    const char* t_sc;                   /* test time scale */
    struct timespec t_s;                /* test start time */
    struct timespec t_e;                /* test end time */
    double t_tak;                       /* test time taken */
    double t_tot;                       /* total test time taken */
} UnitTest;

UNUSED static UnitTest sob_ut = {
    .n_fail = 0,
    .n_pass = 0,
    .n_test = 0,
    .flag = 0,
    .quiet = 0,
    .msg = NULL,

    .t_sc = (SOB_UT_TIMES == 1e3) ? "us" : ((SOB_UT_TIMES == 1e6) ? "ms" : "s"),
    .t_s = {0},
    .t_e = {0},
    .t_tak = 0,
    .t_tot = 0
};

#define UnitTest_fn_def(name)           char* name()
#define _UT_NRES(res)                   sob_ut.n_test++; ((res == NULL) ? (sob_ut.n_pass++) : (sob_ut.n_fail++))
#define _UT_SRES(res)                   ((res == NULL) ? _GREEN("PASS") : _RED("FAIL"))
#define _UT_TSTART()                    clock_gettime(CLOCK_MONOTONIC, &sob_ut.t_s);
#define _UT_TEND()                                                                                                                \
    do {                                                                                                                          \
        clock_gettime(CLOCK_MONOTONIC, &sob_ut.t_e);                                                                              \
        sob_ut.t_tak = ((sob_ut.t_e.tv_sec - sob_ut.t_s.tv_sec) * 1e9 + sob_ut.t_e.tv_nsec - sob_ut.t_s.tv_nsec) / SOB_UT_TIMES;  \
        if (sob_ut.t_e.tv_nsec < sob_ut.t_s.tv_nsec) {                                                                            \
            sob_ut.t_tak += 1;                                                                                                    \
        }                                                                                                                         \
        sob_ut.t_tot += sob_ut.t_tak;                                                                                             \
    } while (0)

#define UnitTest_msg(...)                                                          \
    do {                                                                           \
        char message[64];                                                          \
        snprintf(message, 64, VA_ARG_FIRST(__VA_ARGS__) VA_ARG_REST(__VA_ARGS__)); \
        printf("│  │ " _YELLOW("msg: ") _GREY("%-38s") " │\n", message);           \
    } while (0)

#define UnitTest_ast(test, message)                                   \
    if (!(test)) {                                                    \
        printf("│  │ " _YELLOW("ast: ") _RED("%-38s") " │\n", #test); \
        UnitTest_msg(message);                                        \
        Log_err(message);                                             \
        return message;                                               \
    }

#define UnitTest_add(test)                                                                                                                                       \
    do {                                                                                                                                                         \
        Log_dbg("\n───── Sub: %s", _BLUE(#test));                                                                                                                \
        _UT_TSTART();                                                                                                                                            \
        sob_ut.msg = test();                                                                                                                                     \
        _UT_TEND();                                                                                                                                              \
        _UT_NRES(sob_ut.msg);                                                                                                                                    \
        printf("│  ├── " _MAGENTA("%-2d ") _BLUE("%-18s") _CYAN("%12.4f %2s") " %s │\n", sob_ut.n_test, #test, sob_ut.t_tak, sob_ut.t_sc, _UT_SRES(sob_ut.msg)); \
        Log_dbg("total exec %.3f %2s", sob_ut.t_tak, sob_ut.t_sc);                                                                                               \
        if (sob_ut.msg) return sob_ut.msg;                                                                                                                       \
    } while (0)

#define UnitTest_run(name)                                                                                                                                                                                            \
    int main(int, char* argv[]) {                                                                                                                                                                                     \
        Log_dbg("\n\n───── Run: " _BLUE("%s"), argv[0]);                                                                                                                                                              \
        printf("┌────────────────────────────────────────────────┐\n");                                                                                                                                               \
        printf("│ Test: " _BLUE("%-40s") " │\n", argv[0]);                                                                                                                                                            \
        char* result = name();                                                                                                                                                                                        \
        printf("│ Sum: " _MAGENTA("%-2d ") "[%2d " _GREEN("PASS") " %2d " _RED("FAIL") "] " _CYAN("%12.4f %2s") " %s │\n", sob_ut.n_test, sob_ut.n_pass, sob_ut.n_fail, sob_ut.t_tot, sob_ut.t_sc, _UT_SRES(result)); \
        printf("├────────────────────────────────────────────────┤\n");                                                                                                                                               \
        if (result == NULL) {                                                                                                                                                                                         \
            printf("│ " _CYAN("%-3s ") _BLUE("%-37s ") "%s │\n", "Res", argv[0], _GREEN("PASS"));                                                                                                                     \
        } else {                                                                                                                                                                                                      \
            printf("│ " _CYAN("%-3s ") _BLUE("%-37s ") "%s │\n", "Res", argv[0], _RED("FAIL"));                                                                                                                       \
            printf("│ " _CYAN("%-3s ") _RED("%-42s") " │\n", "Msg", result);                                                                                                                                          \
            printf("│ %-3s %-51s │\n", _CYAN("Log"), _YELLOW("test/tests.log"));                                                                                                                                      \
        }                                                                                                                                                                                                             \
        printf("└────────────────────────────────────────────────┘\n");                                                                                                                                               \
        exit(result != 0);                                                                                                                                                                                            \
    }

// ==================================================================================== //
//                                    sob: argparse (AP)
// ==================================================================================== //

typedef struct {
    char *sarg;
    char *larg;
    char *literal;

    const char *help;

    bool no_val;
    union {
        int i;
        bool b;
        float f;
        char *s;
        void *v;
    } init;
} ArgParserArg;

typedef void (*ArgParser_cmd_fn)(int argc, char *argv[], char *envp[]);


typedef struct ArgParserCmd{
    union {
        // User define command
        struct {
            const char* name;
            const char* desc;
            const char* uasge;

            ArgParserArg *args;
            int   n_args;
            ArgParser_cmd_fn fn;
        };
        // System exec command
        struct {
            CStr * sys_line;
        };
    };

    enum {
        AP_CMD_USER,
        AP_CMD_SYS
    } type;
    
    bool  is_sub;
    struct ArgParserCmd *prev;
    struct ArgParserCmd *next;
} ArgParserCmd;

typedef void (*ArgParser_print_fn)(ArgParserCmd *);

typedef struct {
    const char* prog_path;
    const char* prog_name;
    const char* prog_desc;

    int   n_cmd;
    int   cur_cmd;
    ArgParserCmd cmds[SOB_AP_MSCMD];

    bool  has_global;
    bool  has_subcmd;
    ArgParser_print_fn print_fn;
} ArgParser;

UNUSED static ArgParser sob_ap = {
    .prog_path = NULL,
    .prog_name = NULL,
    .prog_desc = NULL,

    .n_cmd = 0,
    .cur_cmd = 0,

    .has_global = false,
    .has_subcmd = false,
    .print_fn   = NULL
};

#define ArgParser_err_no(N, ...)        Log_err_no(N, "[ArgParser]: " __VA_ARGS__)
#define ArgParser_ast(expr, ...)        Log_ast(expr, "[ArgParser]: " __VA_ARGS__)
#define ArgParser_ast_no(expr, N, ...)  Log_ast_no(expr, N, "[ArgParser]: " __VA_ARGS__)
#define ArgParser_def_args(name)        static ArgParserArg name[]
#define ArgParser_def_fn(name)          void name(UNUSED int argc, UNUSED char *argv[], UNUSED char *envp[])
#define ArgParser_arg_END               { 0 }
#define ArgParser_arg_INPUT             { .sarg = "i", .larg = "input", .init.s = "", .help = "set input file" }
#define ArgParser_arg_OUTPUT            { .sarg = "o", .larg = "output", .init.s = "", .help = "set output file" }
#define ArgParser_cur_cmd               (&sob_ap.cmds[sob_ap.cur_cmd])
#define ArgParser_max_cmd               (sob_ap.cmds[sob_ap.n_cmd])
#define ArgParser_init(Prog_desc, Print_fn)       \
    do {                                          \
        sob_ap.prog_desc = Prog_desc;             \
        if (Print_fn) sob_ap.print_fn = Print_fn; \
    } while (0)

#define ArgParser_use_cmd(Name, Desc, Usage, Fn, Args)                                    \
    do {                                                                                  \
        ArgParser_ast_no(sob_ap.n_cmd < SOB_AP_MSCMD, ERROR_AP_OVER_SUBCMD);              \
        ArgParser_max_cmd.type = AP_CMD_USER;                                             \
        ArgParser_max_cmd.name = (Name == NULL) ? (Fn == NULL ? "" : #Fn) : Name;         \
        ArgParser_max_cmd.desc = (Desc == NULL) ? "" : Desc;                              \
        ArgParser_max_cmd.uasge = (Usage == NULL) ? "" : Usage;                           \
        ArgParser_max_cmd.fn = Fn;                                                        \
        ArgParser_max_cmd.args = Args;                                                    \
        ArgParser_max_cmd.prev = NULL;                                                    \
        ArgParser_max_cmd.next = NULL;                                                    \
        if (strcmp(ArgParser_max_cmd.name, "") == 0) {                                    \
            sob_ap.has_global = true;                                                     \
        } else if (strcmp(ArgParser_max_cmd.name, SOB_AP_GLCMD) == 0) {                   \
            sob_ap.has_global = true;                                                     \
            sob_ap.has_subcmd = true;                                                     \
        } else {                                                                          \
            sob_ap.has_subcmd = true;                                                     \
        }                                                                                 \
        ArgParser_ast_no(sob_ap.has_global || sob_ap.has_subcmd, ERROR_AP_NO_SUBCMD);     \
        int n_args = 0;                                                                   \
        while (1) {                                                                       \
            if (Args[n_args].sarg || Args[n_args].larg) {                                 \
                Args[n_args].help = (Args[n_args].help == NULL) ? "" : Args[n_args].help; \
                n_args++;                                                                 \
            } else {                                                                      \
                break;                                                                    \
            }                                                                             \
        }                                                                                 \
        ArgParser_max_cmd.n_args = n_args;                                                \
        sob_ap.n_cmd++;                                                                   \
    } while (0)

#define ArgParser_sys_CMD(SA)                                                                \
    do {                                                                                     \
        ArgParser_ast_no(sob_ap.n_cmd < SOB_AP_MSCMD, ERROR_AP_OVER_SUBCMD);                 \
        ArgParser_ast_no((SA) != NULL, ERROR_CS_ACCESS_FAIL, "`" _YELLOW_BD("%s") "`", #SA); \
        ArgParser_max_cmd.type = AP_CMD_SYS;                                                 \
        ArgParser_max_cmd.sys_line = (SA);                                                   \
        ArgParser_max_cmd.next = NULL;                                                       \
        ArgParser_max_cmd.prev = NULL;                                                       \
        sob_ap.has_subcmd = true;                                                            \
        sob_ap.n_cmd++;                                                                      \
    } while (0)

#define ArgParser_sys_cmd(S)     \
    do {                         \
        CStr* cmd;               \
        CStrArray_from(&cmd, S); \
        ArgParser_sys_CMD(cmd);  \
    } while (0)

#define ArgParser_print_base_command(Cmd)                                                                                    \
    do {                                                                                                                     \
        fprintf(stderr, "> " _BOLD("%s ") _GREEN_BD("%s") _GREY_IT(" < ... >\n   ") _WHITE_BD_UL("Descr:") _GREY_IT("  %s"), \
                sob_ap.prog_name, Cmd->name, Cmd->desc);                                                                     \
        fprintf(stderr, "\n   " _WHITE_BD_UL("Usage:") _GREY_IT("  %s\n"), Cmd->uasge);                                      \
        for (int i = 0; i < Cmd->n_args; i++) {                                                                              \
            if ((Cmd)->type == AP_CMD_USER) {                                                                                \
                fprintf(stderr, "       " _RED("%s%s") "  %s%-10s" _GREY_IT("%s\n"), SOB_AP_SFLAG, Cmd->args[i].sarg,        \
                        SOB_AP_LFLAG, Cmd->args[i].larg, Cmd->args[i].help);                                                 \
            } else {                                                                                                         \
            }                                                                                                                \
        }                                                                                                                    \
        fprintf(stderr, "\n");                                                                                               \
    } while (0)

#define ArgParser_print_help_command(Cmd)                                                                       \
    if ((Cmd)->type == AP_CMD_USER) {                                                                           \
        fprintf(stderr, _GREEN_BD(" %8s") _GREY_IT("  %s\n"), (Cmd)->name, (Cmd)->desc);                        \
        for (int i = 0; i < (Cmd)->n_args; i++) {                                                               \
            if (i >= SOB_AP_NFLAG) {                                                                            \
                fprintf(stderr, "       " _RED("%s%s") "  %s%-10s" _GREY_IT("%s\n"),                            \
                        SOB_AP_SFLAG, "h", SOB_AP_LFLAG, "help", "Get more info ...");                          \
                break;                                                                                          \
            }                                                                                                   \
            fprintf(stderr, "       " _RED("%s%s") "  %s%-10s" _GREY_IT("%s\n"),                                \
                    SOB_AP_SFLAG, (Cmd)->args[i].sarg, SOB_AP_LFLAG, (Cmd)->args[i].larg, (Cmd)->args[i].help); \
        }                                                                                                       \
        fprintf(stderr, "\n");                                                                                  \
    } else if ((Cmd)->type == AP_CMD_SYS) {                                                                     \
        fprintf(stderr, _CYAN_BD(" %8s") "  ", (Cmd)->sys_line[0]);                                             \
        CStr* cur_line = (Cmd)->sys_line + 1;                                                                   \
        while (*cur_line) {                                                                                     \
            fprintf(stderr, "%s ", *cur_line);                                                                  \
            cur_line++;                                                                                         \
        }                                                                                                       \
        fprintf(stderr, "\n\n");                                                                                \
    }

#define ArgParser_print_parser()                                                                  \
    do {                                                                                          \
        fprintf(stderr, "> " _BOLD("%s ") _WHITE(": %s\n "), sob_ap.prog_name, sob_ap.prog_desc); \
        if (sob_ap.n_cmd > 1) {                                                                   \
            fprintf(stderr, _WHITE_BD_UL("Usage:\n"));                                            \
            for (int i = 0; i < sob_ap.n_cmd; i++) {                                              \
                ArgParser_print_help_command(&sob_ap.cmds[i]);                                    \
            }                                                                                     \
        } else {                                                                                  \
            sob_ap.print_fn(ArgParser_cur_cmd);                                                   \
        }                                                                                         \
    } while (0)

#define ArgParser_print_command()                        \
    if (!sob_ap.print_fn) {                              \
        ArgParser_print_base_command(ArgParser_cur_cmd); \
    } else {                                             \
        sob_ap.print_fn(ArgParser_cur_cmd);              \
    }

#define _ArgParser_cmd(Argc, Argv)                                                                      \
    do {                                                                                                \
        if (Argc > 0) {                                                                                 \
            if (strcmp(Argv[0], "-h") == 0 || strcmp(Argv[0], "--help") == 0) {                         \
                ArgParser_print_command();                                                              \
                exit(ERROR_SOB_NONE);                                                                   \
            }                                                                                           \
        }                                                                                               \
        bool is_arg_name = true;                                                                        \
        int count = 0;                                                                                  \
        int sflag_len = strlen(SOB_AP_SFLAG);                                                           \
        int lflag_len = strlen(SOB_AP_LFLAG);                                                           \
        ArgParserArg* arg = NULL;                                                                       \
        char* arg_name = NULL;                                                                          \
        while (count < Argc) {                                                                          \
            char* need_parse = Argv[count];                                                             \
            int need_parse_len = strlen(need_parse);                                                    \
            bool is_short = (strncmp(need_parse, SOB_AP_SFLAG, MIN(sflag_len, need_parse_len)) == 0);   \
            bool is_long = (strncmp(need_parse, SOB_AP_LFLAG, MIN(lflag_len, need_parse_len)) == 0);    \
            bool is_file = false;                                                                       \
            FILE* fp = fopen(need_parse, "r");                                                          \
            if (fp) {                                                                                   \
                is_file = true;                                                                         \
                fclose(fp);                                                                             \
            }                                                                                           \
            if (is_short || is_long) {                                                                  \
                is_arg_name = true;                                                                     \
                const char* prefix;                                                                     \
                if (is_short) {                                                                         \
                    prefix = SOB_AP_SFLAG;                                                              \
                } else {                                                                                \
                    prefix = SOB_AP_LFLAG;                                                              \
                }                                                                                       \
                arg_name = need_parse + strlen(prefix);                                                 \
                bool exist = false;                                                                     \
                for (int i = 0; i < ArgParser_cur_cmd->n_args; i++) {                                   \
                    arg = &(ArgParser_cur_cmd->args[i]);                                                \
                    bool short_exist = strcmp(arg->sarg, arg_name) == 0;                                \
                    bool long_exist = strcmp(arg->larg, arg_name) == 0;                                 \
                    if (short_exist || long_exist) {                                                    \
                        exist = true;                                                                   \
                        break;                                                                          \
                    }                                                                                   \
                }                                                                                       \
                if (!exist) {                                                                           \
                    ArgParser_err_no(ERROR_AP_NO_EXIST_ARG);                                            \
                    exit(ERROR_AP_NO_EXIST_ARG);                                                        \
                }                                                                                       \
            } else {                                                                                    \
                is_arg_name = false;                                                                    \
            }                                                                                           \
            if (arg == NULL) {                                                                          \
                if (is_file) {                                                                          \
                    bool exist = false;                                                                 \
                    for (int i = 0; i < ArgParser_cur_cmd->n_args; i++) {                               \
                        arg = &(ArgParser_cur_cmd->args[i]);                                            \
                        bool short_exist = strcmp("i", need_parse) == 0;                                \
                        bool long_exist = strcmp("input", need_parse) == 0;                             \
                        if (short_exist || long_exist) {                                                \
                            exist = true;                                                               \
                            break;                                                                      \
                        }                                                                               \
                    }                                                                                   \
                    if (!exist) {                                                                       \
                        ArgParser_err_no(ERROR_AP_NO_EXIST_ARG);                                        \
                        exit(ERROR_AP_NO_EXIST_ARG);                                                    \
                    }                                                                                   \
                } else {                                                                                \
                    ArgParser_err_no(ERROR_AP_LOST_ARG_FLAG, "`" _YELLOW_BD("%s") "`", need_parse);     \
                    exit(ERROR_AP_LOST_ARG_FLAG);                                                       \
                }                                                                                       \
            }                                                                                           \
            if (!is_arg_name && arg->no_val) {                                                          \
                ArgParser_err_no(ERROR_AP_EXTRA_VAL);                                                   \
                exit(ERROR_AP_EXTRA_VAL);                                                               \
            }                                                                                           \
            if (!is_arg_name && !arg->no_val) {                                                         \
                arg->literal = need_parse;                                                              \
                arg = NULL;                                                                             \
            }                                                                                           \
            if (is_arg_name && arg->no_val) {                                                           \
                arg->init.b = true;                                                                     \
            }                                                                                           \
            count++;                                                                                    \
        }                                                                                               \
        if (arg && !arg->no_val) {                                                                      \
            ArgParser_err_no(ERROR_AP_LOST_ARG_VAL, "`" _YELLOW_BD(SOB_AP_SFLAG "%s") "`", arg->sarg);  \
            exit(ERROR_AP_LOST_ARG_VAL);                                                                \
        }                                                                                               \
    } while (0)

#define ArgParser_run(Argc, Argv, Envp)                                                                                         \
    do {                                                                                                                        \
        int argc_copy = Argc;                                                                                                   \
        char** argv_copy = Argv;                                                                                                \
        sob_ap.prog_path = Argv[0];                                                                                             \
        sob_ap.prog_name = strrchr(sob_ap.prog_path, '/');                                                                      \
        if (sob_ap.prog_name) {                                                                                                 \
            sob_ap.prog_name++;                                                                                                 \
        } else {                                                                                                                \
            sob_ap.prog_name = sob_ap.prog_path;                                                                                \
        }                                                                                                                       \
        if (Argc > 1) {                                                                                                         \
            if (strcmp(Argv[1], "-h") == 0 || strcmp(Argv[1], "--help") == 0) {                                                 \
                ArgParser_print_parser();                                                                                       \
                exit(ERROR_SOB_NONE);                                                                                           \
            }                                                                                                                   \
        }                                                                                                                       \
        Argc--;                                                                                                                 \
        Argv++;                                                                                                                 \
        if (sob_ap.has_subcmd) {                                                                                                \
            char* subcmd;                                                                                                       \
            bool exist = false;                                                                                                 \
            if (!Argc && !sob_ap.has_global) {                                                                                  \
                ArgParser_print_parser();                                                                                       \
                ArgParser_err_no(ERROR_AP_NO_EXIST_SUBCMD, "`" _YELLOW_BD("all") "`");                                          \
                exit(ERROR_AP_NO_EXIST_SUBCMD);                                                                                 \
            } else if (Argc && sob_ap.has_global) {                                                                             \
                subcmd = Argv[0];                                                                                               \
                for (int i = 0; i < sob_ap.n_cmd; i++) {                                                                        \
                    if (subcmd && sob_ap.cmds[i].type == AP_CMD_USER && strcmp(subcmd, sob_ap.cmds[i].name) == 0) {             \
                        sob_ap.cmds[i].is_sub = true;                                                                           \
                        sob_ap.cur_cmd = i;                                                                                     \
                        exist = true;                                                                                           \
                        break;                                                                                                  \
                    } else if (subcmd && sob_ap.cmds[i].type == AP_CMD_SYS) {                                                   \
                        CStr prog = CStrArray_get(sob_ap.cmds[i].sys_line, 0) ? CStrArray_get(sob_ap.cmds[i].sys_line, 0) : ""; \
                        if (strcmp(subcmd, prog) == 0) {                                                                        \
                            sob_ap.cmds[i].is_sub = true;                                                                       \
                            sob_ap.cur_cmd = i;                                                                                 \
                            exist = true;                                                                                       \
                            break;                                                                                              \
                        }                                                                                                       \
                    }                                                                                                           \
                }                                                                                                               \
                if (!exist) {                                                                                                   \
                    ArgParser_err_no(ERROR_AP_NO_EXIST_SUBCMD, "`" _YELLOW_BD("%s") "`", subcmd);                               \
                    exit(ERROR_AP_NO_EXIST_SUBCMD);                                                                             \
                }                                                                                                               \
            } else if (!Argc && sob_ap.has_global) {                                                                            \
                subcmd = SOB_AP_GLCMD;                                                                                          \
            } else {                                                                                                            \
                subcmd = Argv[0];                                                                                               \
                for (int i = 0; i < sob_ap.n_cmd; i++) {                                                                        \
                    if (strcmp(subcmd, sob_ap.cmds[i].name) == 0) {                                                             \
                        sob_ap.cmds[i].is_sub = true;                                                                           \
                        sob_ap.cur_cmd = i;                                                                                     \
                        exist = true;                                                                                           \
                        break;                                                                                                  \
                    }                                                                                                           \
                }                                                                                                               \
                if (!exist) {                                                                                                   \
                    ArgParser_err_no(ERROR_AP_NO_EXIST_SUBCMD, "`" _YELLOW_BD("%s") "`", subcmd);                               \
                    exit(ERROR_AP_NO_EXIST_SUBCMD);                                                                             \
                }                                                                                                               \
            }                                                                                                                   \
        } else {                                                                                                                \
            sob_ap.cur_cmd = 0;                                                                                                 \
        }                                                                                                                       \
        Argc -= (sob_ap.has_subcmd ? 1 : 0);                                                                                    \
        Argv += (sob_ap.has_subcmd ? 1 : 0);                                                                                    \
        _ArgParser_cmd(Argc, Argv);                                                                                             \
        ArgParserCmd* cur_cmd = (ArgParserCmd*)ArgParser_cur_cmd;                                                               \
        while (cur_cmd) {                                                                                                       \
            if (cur_cmd->type == AP_CMD_USER) {                                                                                 \
                cur_cmd->fn(argc_copy, argv_copy, Envp);                                                                        \
            } else if (cur_cmd->type == AP_CMD_SYS) {                                                                           \
                SobPid PID;                                                                                                     \
                FORK(PID);                                                                                                      \
                CStr* cmd_line = cur_cmd->sys_line;                                                                             \
                if (cmd_line) {                                                                                                 \
                    CMD(PID, cmd_line);                                                                                         \
                }                                                                                                               \
            }                                                                                                                   \
            cur_cmd = cur_cmd->next;                                                                                            \
        }                                                                                                                       \
    } while (0)

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // _SOB_SOB_H_
