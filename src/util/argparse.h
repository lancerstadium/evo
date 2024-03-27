/**
 * @file argparse.h
 * @author lancer (you@domain.com)
 * @brief 命令行解析库头文件
 * @version 0.1
 * @date 2023-12-25
 *
 * @copyright Copyright (c) 2023
 * @note 参考项目：[Github | argparse](https://github.com/dwpeng/argparse)
 * @note
 * # 参数解析步骤
 * 1. Step1: 使用宏 `ap_def_args` 定义参数
 * ```c
 * #include "argparse.h"
 * 
 * ap_def_args(test_args) = {
 *   {.short_arg = "o", .long_arg = "output", .init.s = "./test", .help = "set output path"},
 *   {.short_arg = "q", .long_arg = "quiet" , .init.b = true    , .help = "quiet run"},
 *   {.short_arg = "d", .long_arg = "debug" , .init.b = false   , .help = "debug mode"},
 *   AP_END_ARG
 * };
 * ```
 *
 * 2. Step2: 使用宏 `ap_def_callback` 定义回调函数
 * ```c
 * ap_def_callback(test_callback) {
 *   ap_arg_t *arg = ap_get("output");
 *   if (!arg->value) {
 *     printf("no value. init: %s\n", arg->init.s);
 *   }
 *   else {
 *     printf("option output: %s\n", arg->value);
 *   }
 *   printf("option quiet: %d\n", ap_get("quiet")->init.b);
 *   printf("option debug: %d\n", ap_get("debug")->init.b);
 * }
 * ```
 *
 * 3. Step3: 使用宏 `ap_init_parser` 初始化解析器
 * ```c
 * ap_init_parser("uemu - a simple emulator", NULL);
 * ```
 *
 * 4. Step4: 使用宏 `ap_add_command` 添加命令
 * ```c
 * ap_add_command("test", "Print `Hello, World!`.", "This is usage.", test_callback, test_args);
 * ```
 * @attention 添加命令名为 `default` 时，是设置主命令：
 *
 * 5. Step5: 使用宏 `ap_add_command` 解析命令
 * ```c
 * ap_do_parser(argc, argv, envp);
 * ```
 * 6. Step6：示例
 * ```c
 * void arg_parse(int argc, char *argv[], char *envp[])
 * {
 *  // Step3: 初始化解析器
 *  ap_init_parser("uemu - a simple emulator", NULL);
 *  // Step4: 添加命令
 *  ap_add_command("default", "Default: print `Hello, World!`", "uemu test", hello_callback, help_args);
 *  ap_add_command("help", "This is description.", "This is usage.", help_callback, help_args); 
 *  ap_add_command("debug", "Enter debug mode.", "This is usage.", debug_callback, debug_args);
 *  ap_add_command("test", "Unit test", "This is usage.", test_callback, test_args);
 *  // Step5: 开始解析
 *  ap_do_parser(argc, argv, envp);
 * }
 * 
 * ```
 */

#ifndef ARGPARSE_H
#define ARGPARSE_H

// ==================================================================== //
//                                Include
// ==================================================================== //


#ifdef __cplusplus
	extern "C" {
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "color.h"



// ==================================================================== //
//                                  Define
// ==================================================================== //


/* 解析器包含的子命令数 */
#ifndef AP_MAX_NCOMMAND
#define AP_MAX_NCOMMAND 10
#endif

/* 长参数名前的flag */
#ifndef AP_LONG_FLAG
#define AP_LONG_FLAG "--"
#endif

/* 短参数名前的flag */
#ifndef AP_SHORT_FLAG
#define AP_SHORT_FLAG "-"
#endif

#define AP_DEFAULT_COMMAND "default"

#define AP_END_ARG \
    {          \
        0      \
    }

#define AP_INPUT_ARG \
    {   \
        .short_arg = "i",  \
        .long_arg = "input", \
        .init.s = "", \
        .help = "set input file" \
    }

#define ap_min(a, b) ((a) > (b) ? (b) : (a))
#define ap_max(a, b) ((a) > (b) ? (a) : (b))

#define NOW_CMD (&pap->commands[pap->command_pos])

/* 错误提示 */
#define ERROR_MSG _red(" [ERROR]")
#define ap_err(msg) (ERROR_MSG ": " msg "\n")

// 命令冲突
#ifndef ERROR_COMMAND_CONFLICT
#define ERROR_COMMAND_CONFLICT ap_err("Conflict.")
#endif

// 没有传递子命令
#ifndef ERROR_NO_SUBCOMMAND
#define ERROR_NO_SUBCOMMAND ap_err("Pass a subcommand.")
#endif

// 没有传递参数值
#ifndef ERROR_LOST_ARG_VALUE
#define ERROR_LOST_ARG_VALUE ap_err("\"%s\" lost arg value.")
#endif

#ifndef ERROR_DONOT_NEED_VALUE
#define ERROR_DONOT_NEED_VALUE ap_err("\"%s\" does not need arg value.")
#endif

#ifndef ERROR_ARG_NOT_EXIST
#define ERROR_ARG_NOT_EXIST ap_err("Arg name \"%s\" does not exist.")
#endif

#ifndef ERROR_LOST_ARG_NAME
#define ERROR_LOST_ARG_NAME ap_err("Except a arg name, but got \"%s\".")
#endif

#ifndef ERROR_SUBCOMMAND_NOT_EXIST
#define ERROR_SUBCOMMAND_NOT_EXIST ap_err("Subcommand %s does not exist.")
#endif


// ==================================================================== //
//                              Sturct: Argparser
// ==================================================================== //

/* 是否跟随参数 */
typedef enum {
    ap_YES = 0,
    ap_NO
} ArgValue;

typedef struct {
    /* 单短线参数名 */
    char *short_arg;
    /* 双短线参数名 */
    char *long_arg;
    /* 是否跟随参数值 */
    ArgValue arg_have_value;
    /* 传递值 */
    char *value;
    /* 初始值 */
    union {
        int i;
        bool b;
        float f;
        char *s;
        void *v;
    } init;
    /* 参数的说明 */
    char *help;
} ap_arg_t;


/* command对应的回调函数 */
typedef void (*callback_t)(int argc, char *argv[], char *envp[]);

typedef struct
{
    /* 命令名
     * - 全局命令为NULL
     * - 子命令时，为子命令名
     **/
    char *command;
    /* 对命令的描述 */
    char *description;
    /* 命令的用法 */
    char *usage;
    /* 是否是子命令 */
    int subcommand;
    /* 参数个数 */
    int nargs;
    /* 命令的回调函数 */
    callback_t callback;
    /* 存储的参数 */
    ap_arg_t *args;
} ap_command_t;

/* 打印command的回调函数 */
typedef void (*print_ap_command_t)(ap_command_t *);

struct argparse_t
{
    /* 程序开始运行打印的内容
     * 可以是logo之类的
     **/
    char *print;
    /* 命令个数 */
    int ncommand;
    /* 存储的命令 */
    ap_command_t commands[AP_MAX_NCOMMAND];
    /* 当前命令行使用的command命令 */
    int command_pos;
    /* 自定义打印command参数 */
    print_ap_command_t print_command;
    /* 程序名 */
    char *prog_name;    
    /* 程序名去除路径 */         
    char *prog_name_without_path;
    /* 有全局命令 */
    int have_global;
    /* 有子命令 */
    int have_subcommand;
};


// ==================================================================== //
//                            Declare API: Argparse
// ==================================================================== //


/* 定义参数用到的宏 */
#define ap_def_args(name) static ap_arg_t name[]
#define ap_def_callback(name) void name(int argc, char *argv[], char *envp[])

/**
 * @brief 初始化解析器
 *
 * @param print_message
 * @param print_command
 */
void ap_init_parser(char *print_message, print_ap_command_t print_command);

/**
 * @brief 添加一个子命令
 *
 * @param command
 * @param description
 * @param usage
 * @param callback
 * @param args
 */
void ap_add_command(char *command, char *description, char *usage, callback_t callback, ap_arg_t *args);

/**
 * @brief 根据参数名获取参数值
 *
 * @param arg_name
 * @return void*
 */
ap_arg_t* ap_get(char *arg_name);

/**
 * @brief 打印command
 * @param c
 */
static inline void ap_default_print_command(ap_command_t *c);

/**
 * @brief 打印 command 简易版
 * @param c
 */
static inline void ap_default_print_base_command(ap_command_t *c);

/**
 * @brief 打印解析器
 */
static inline void ap_print_parser(void);

static inline void ap_print_command(void);

/**
 * @brief 解析命令行
 *
 * @param argc
 * @param argv
 */
void ap_do_parser(int argc, char *argv[], char *envp[]);

#ifdef __cplusplus
}
#endif

#endif
