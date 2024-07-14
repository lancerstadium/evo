/**
 * =================================================================================== //
 * @file sob.h
 * @author lancerstadium (lancerstadium@163.com)
 * @brief Super No Build Toolkit
 * @version 0.0.4
 * @date 2024-06-13
 * @copyright Copyright (c) 2024
 * =================================================================================== //
 */

#ifndef __SOB_SOB_H__
#define __SOB_SOB_H__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ==================================================================================== //
//                                       Include
// ==================================================================================== //

#include <time.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdbool.h>


// ==================================================================================== //
//                                    sob: SOB Config (SOB)
// ==================================================================================== //

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

#define MAX(a, b)           ({typeof(a) _amin = (a); typeof(b) _bmin = (b); (void)(&_amin == &_bmin); _amin < _bmin ? _amin : _bmin;})
#define MIN(a, b)           ({typeof(a) _amax = (a); typeof(b) _bmax = (b); (void)(&_amax == &_bmax); _amax > _bmax ? _amax : _bmax;})
#define CLAMP(v, a, b)		MIN(MAX(a, v), b)

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
#define CONCAT5(a, b, c, d, e) CONCAT(CONCAT4(a, b, c, d), e)
#define CONCAT6(a, b, c, d, e, f) CONCAT(CONCAT5(a, b, c, d, e), f)
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

#define LOAD_LE_1(buf) ((size_t) *(uint8_t*) (buf))
#define LOAD_LE_2(buf) (LOAD_LE_1(buf) | LOAD_LE_1((uint8_t*) (buf) + 1)<<8)
#define LOAD_LE_4(buf) (LOAD_LE_2(buf) | LOAD_LE_2((uint8_t*) (buf) + 2)<<16)
#define UBFX(val, start, end) (((val) >> start) & ((1 << (end - start + 1)) - 1))
#define SBFXIZ(val, start, end, shl) ((struct { long v: end-start+1+shl; }) {UBFX(val, start, end)<<shl}.v)

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


#ifdef __GNUC__
#define LIKELY(x) __builtin_expect((x), 1)
#define UNLIKELY(x) __builtin_expect((x), 0)
#define ASSUME(x) do { if (!(x)) __builtin_unreachable(); } while (0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define ASSUME(x) ((void) 0)
#endif

#undef offsetof
#ifdef __compiler_offsetof
#define offsetof(TYPE, MEMBER) __compiler_offsetof(TYPE,MEMBER)
#else
#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)
#endif

#define container_of(ptr, type, member) ({			\
	const typeof(((type *)0)->member) * __mptr = (ptr);	\
	(type *)((char *)__mptr - offsetof(type, member)); })


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
    LOGLEVEL_PURE,
    LOGLEVEL_TODO,
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
    { LOGLEVEL_PURE, "PURE", ANSI_FG_WHITE          },
    { LOGLEVEL_PURE, "TODO", ANSI_FGB_YELLOW        },
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

#define Log_errno                   (sob_logger.no == ERROR_SOB_NONE ? (strerror(errno)) : sob_logger.error[sob_logger.no].msg)
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

#define TODO()                      Log_msg(LOGLEVEL_TODO, "TODO: %s", __func__)
#define Log(...)                    Log_msg(LOGLEVEL_PURE, ##__VA_ARGS__)
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
    int   cur_arg;
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
    .cur_arg = -1,

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
#define ArgParser_cur_arg               ((sob_ap.cur_arg == -1) ? NULL : (&((ArgParser_cur_cmd)->args[sob_ap.cur_arg])))
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


#define ArgParser_eq(ARG, NAME)  ((strcmp(NAME, (ARG).sarg) == 0) || (strcmp(NAME, (ARG).larg) == 0))

#define ArgParser_get(NAME)                               \
    do {                                                  \
        if ((ArgParser_cur_cmd) != NULL) {                \
            int n_args = ArgParser_cur_cmd->n_args;       \
            ArgParserArg* args = ArgParser_cur_cmd->args; \
            sob_ap.cur_arg = -1;                          \
            for (int i = 0; i < n_args; i++) {            \
                if (ArgParser_eq(args[i], NAME)) {        \
                    sob_ap.cur_arg = i;                   \
                    break;                                \
                }                                         \
            }                                             \
        }                                                 \
    } while (0)

#define ArgParser_print_base_command(Cmd)                                                                              \
    do {                                                                                                               \
        fprintf(stderr, "> " _BOLD("%s ") _GREEN_BD("%s") _GREY(" < ... >\n   ") _WHITE_BD_UL("Descr:") _GREY("  %s"), \
                sob_ap.prog_name, Cmd->name, Cmd->desc);                                                               \
        fprintf(stderr, "\n   " _WHITE_BD_UL("Usage:") _GREY("  %s\n"), Cmd->uasge);                                   \
        for (int i = 0; i < Cmd->n_args; i++) {                                                                        \
            if ((Cmd)->type == AP_CMD_USER) {                                                                          \
                fprintf(stderr, "       " _RED("%s%s") "  %s%-10s" _GREY("%s\n"), SOB_AP_SFLAG, Cmd->args[i].sarg,     \
                        SOB_AP_LFLAG, Cmd->args[i].larg, Cmd->args[i].help);                                           \
            } else {                                                                                                   \
            }                                                                                                          \
        }                                                                                                              \
        fprintf(stderr, "\n");                                                                                         \
    } while (0)

#define ArgParser_print_help_command(Cmd)                                                 \
    if ((Cmd)->type == AP_CMD_USER) {                                                     \
        fprintf(stderr, _GREEN_BD("    %-8s") _GREY("  %s\n"), (Cmd)->name, (Cmd)->desc); \
    }

#define ArgParser_print_options(Cmd)                                                                            \
    if ((Cmd)->type == AP_CMD_USER) {                                                                           \
        for (int i = 0; i < (Cmd)->n_args; i++) {                                                               \
            if (i >= SOB_AP_NFLAG) {                                                                            \
                fprintf(stderr, "       " _RED("%s%s") "  %s%-10s" _GREY("%s\n"),                               \
                        SOB_AP_SFLAG, "h", SOB_AP_LFLAG, "help", "Get more info ...");                          \
                break;                                                                                          \
            }                                                                                                   \
            fprintf(stderr, "       " _RED("%s%s") "  %s%-10s" _GREY("%s\n"),                                   \
                    SOB_AP_SFLAG, (Cmd)->args[i].sarg, SOB_AP_LFLAG, (Cmd)->args[i].larg, (Cmd)->args[i].help); \
        }                                                                                                       \
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

#define _ArgParser_cmd(Argc, Argv)                                                                                  \
    do {                                                                                                            \
        if (Argc > 0) {                                                                                             \
            if (strcmp(Argv[0], "-h") == 0 || strcmp(Argv[0], "--help") == 0) {                                     \
                ArgParser_print_command();                                                                          \
                exit(ERROR_SOB_NONE);                                                                               \
            }                                                                                                       \
        }                                                                                                           \
        bool is_arg_name = true;                                                                                    \
        int count = 0;                                                                                              \
        int sflag_len = strlen(SOB_AP_SFLAG);                                                                       \
        int lflag_len = strlen(SOB_AP_LFLAG);                                                                       \
        ArgParserArg* arg = NULL;                                                                                   \
        char* arg_name = NULL;                                                                                      \
        while (count < Argc) {                                                                                      \
            char* need_parse = Argv[count];                                                                         \
            int need_parse_len = strlen(need_parse);                                                                \
            bool is_long = (strncmp(need_parse, SOB_AP_LFLAG, MIN(lflag_len, need_parse_len)) == 0);                \
            bool is_short = (!is_long) && (strncmp(need_parse, SOB_AP_SFLAG, MIN(sflag_len, need_parse_len)) == 0); \
            bool is_file = false;                                                                                   \
            FILE* fp = fopen(need_parse, "r");                                                                      \
            if (fp) {                                                                                               \
                is_file = true;                                                                                     \
                fclose(fp);                                                                                         \
            }                                                                                                       \
            if (is_short || is_long) {                                                                              \
                is_arg_name = true;                                                                                 \
                const char* prefix;                                                                                 \
                if (is_short) {                                                                                     \
                    prefix = SOB_AP_SFLAG;                                                                          \
                } else {                                                                                            \
                    prefix = SOB_AP_LFLAG;                                                                          \
                }                                                                                                   \
                arg_name = need_parse + strlen(prefix);                                                             \
                bool exist = false;                                                                                 \
                for (int i = 0; i < ArgParser_cur_cmd->n_args; i++) {                                               \
                    arg = &(ArgParser_cur_cmd->args[i]);                                                            \
                    bool short_exist = strcmp(arg->sarg, arg_name) == 0;                                            \
                    bool long_exist = strcmp(arg->larg, arg_name) == 0;                                             \
                    if (short_exist || long_exist) {                                                                \
                        exist = true;                                                                               \
                        break;                                                                                      \
                    }                                                                                               \
                }                                                                                                   \
                if (!exist) {                                                                                       \
                    ArgParser_err_no(ERROR_AP_NO_EXIST_ARG, "`" _YELLOW_BD("%s") "`", arg_name);                    \
                    exit(ERROR_AP_NO_EXIST_ARG);                                                                    \
                }                                                                                                   \
            } else {                                                                                                \
                is_arg_name = false;                                                                                \
            }                                                                                                       \
            if (arg == NULL) {                                                                                      \
                if (is_file) {                                                                                      \
                    bool exist = false;                                                                             \
                    for (int i = 0; i < ArgParser_cur_cmd->n_args; i++) {                                           \
                        arg = &(ArgParser_cur_cmd->args[i]);                                                        \
                        bool short_exist = strcmp("i", need_parse) == 0;                                            \
                        bool long_exist = strcmp("input", need_parse) == 0;                                         \
                        if (short_exist || long_exist) {                                                            \
                            exist = true;                                                                           \
                            break;                                                                                  \
                        }                                                                                           \
                    }                                                                                               \
                    if (!exist) {                                                                                   \
                        ArgParser_err_no(ERROR_AP_NO_EXIST_ARG, "`" _YELLOW_BD("%s") "`", need_parse);              \
                        exit(ERROR_AP_NO_EXIST_ARG);                                                                \
                    }                                                                                               \
                } else {                                                                                            \
                    ArgParser_err_no(ERROR_AP_LOST_ARG_FLAG, "`" _YELLOW_BD("%s") "`", need_parse);                 \
                    exit(ERROR_AP_LOST_ARG_FLAG);                                                                   \
                }                                                                                                   \
            }                                                                                                       \
            if (!is_arg_name && arg->no_val) {                                                                      \
                ArgParser_err_no(ERROR_AP_EXTRA_VAL);                                                               \
                exit(ERROR_AP_EXTRA_VAL);                                                                           \
            }                                                                                                       \
            if (!is_arg_name && !arg->no_val) {                                                                     \
                arg->literal = need_parse;                                                                          \
                arg = NULL;                                                                                         \
            }                                                                                                       \
            if (is_arg_name && arg->no_val) {                                                                       \
                arg->init.b = true;                                                                                 \
            }                                                                                                       \
            count++;                                                                                                \
        }                                                                                                           \
        if (arg && !arg->no_val) {                                                                                  \
            ArgParser_err_no(ERROR_AP_LOST_ARG_VAL, "`" _YELLOW_BD(SOB_AP_SFLAG "%s") "`", arg->sarg);              \
            exit(ERROR_AP_LOST_ARG_VAL);                                                                            \
        }                                                                                                           \
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
            }                                                                                                                   \
            cur_cmd = cur_cmd->next;                                                                                            \
        }                                                                                                                       \
    } while (0)

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __SOB_SOB_H__
