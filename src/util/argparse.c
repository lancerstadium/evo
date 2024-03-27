/**
 * @file argparse.c
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */


// ==================================================================== //
//                                Include
// ==================================================================== //

#include "argparse.h"


// ==================================================================== //
//                                Data: argparse
// ==================================================================== //

static struct argparse_t ap = {
    .ncommand = 0,
    .command_pos = 0,
    .prog_name = NULL,
    .prog_name_without_path = NULL,
    .have_global = 0,
    .have_subcommand = 0
};
static struct argparse_t *pap = &ap;


// ==================================================================== //
//                            Func API: Argparse
// ==================================================================== //

void ap_init_parser(char *print_message, print_ap_command_t print_command) {
    pap->print = print_message;
    if (print_command != NULL)
    {
        pap->print_command = print_command;
    }
}


void ap_add_command(char *command, char *description, char *usage, callback_t callback, ap_arg_t *args) {
    if (command == NULL)
    {
        ap.have_global = 1;
    }
    else if (strcmp(command, AP_DEFAULT_COMMAND))
    {
        ap.have_global = 1;
        ap.have_subcommand = 1;
    }
    else
    {
        ap.have_subcommand = 1;
    }
    /* 全局命令和子命令有且只能存在一种
     * 要么有一个全局命令
     * 要么存在多个子命令
     **/

    // if (ap.have_global && ap.have_subcommand)
    // {
    //     fprintf(stderr, ERROR_COMMAND_CONFLICT);
    //     exit(1);
    // }
    if (!(ap.have_global || ap.have_subcommand))
    {
        fprintf(stderr, ERROR_NO_SUBCOMMAND);
        exit(1);
    }

    if (pap->ncommand > AP_MAX_NCOMMAND - 1)
    {
        fprintf(stderr, _red(" [ERROR]:") "Too many commands. Change AP_MAX_NCOMMAND bigger.\n");
        exit(1);
    }

    pap->commands[pap->ncommand].command = command;
    pap->commands[pap->ncommand].description = (char *)malloc(strlen(description) + 1);
    strcpy(pap->commands[pap->ncommand].description, description == NULL ? "" : description);
    pap->commands[pap->ncommand].usage = (char *)malloc(strlen(usage) + 1);
    strcpy(pap->commands[pap->ncommand].usage, usage == NULL ? "" : usage);
    pap->commands[pap->ncommand].callback = callback;
    pap->commands[pap->ncommand].args = args;
    // 统计当前command有多少参数
    int nargs = 0;
    while (1)
    {
        if (args[nargs].long_arg || args[nargs].short_arg)
        {
            if (args[nargs].help == NULL)
            {
                args[nargs].help = "";
            }
            nargs++;
        }
        else
        {
            break;
        }
    }
    pap->commands[pap->ncommand].nargs = nargs;
    pap->ncommand++;
}


/**
 * @brief 判断当前参数名是否与command中的参数名相同
 *
 * @param arg_name
 * @param arg
 * @return int
 */
static inline int _is_eq(char *arg_name, ap_arg_t arg) {
    char *short_arg = arg.short_arg;
    char *long_arg = arg.long_arg;
    if (strcmp(arg_name, long_arg) == 0 || strcmp(arg_name, short_arg) == 0)
    {
        return 1;
    }
    return 0;
}


ap_arg_t * ap_get(char *arg_name) {
    if ((NOW_CMD) == NULL)
    {
        return NULL;
    }
    ap_arg_t *temp_ap = NULL;
    int nargs = (NOW_CMD)->nargs;
    ap_arg_t *args = (NOW_CMD)->args;
    for (int i = 0; i < nargs; i++)
    {
        if (_is_eq(arg_name, args[i]))
        {
            temp_ap = &args[i];
            break;
        }
    }
    return temp_ap;
}



static inline void ap_default_print_command(ap_command_t *c) {
    fprintf(stderr, "\n> " _bold("%s ") _bgreen("%s") _blacki(" < ... >\n   ") _boldu("Descr:") _blacki("  %s"), ap.prog_name_without_path, c->command, c->description);
    fprintf(stderr, "\n   " _boldu("Usage:") _blacki("  %s\n"), c->usage);
    for (size_t i = 0; i < c->nargs; i++)
    {
        fprintf(
            stderr,
            "       " _red("%s%s") "  %s%-10s" _blacki("%s\n"),
            AP_SHORT_FLAG,
            c->args[i].short_arg,
            AP_LONG_FLAG,
            c->args[i].long_arg,
            c->args[i].help);
    }
    fprintf(stderr, "\n");
}


static inline void ap_default_print_base_command(ap_command_t *c) {
    fprintf(stderr, _bgreen(" %8s") _blacki("  %s\n"), c->command, c->description);

    for (size_t i = 0; i < c->nargs; i++)
    {
        if (i >= 2)
        {
            fprintf(
                stderr,
                "       " _red("%s%s") "  %s%-10s" _blacki("%s\n"),
                AP_SHORT_FLAG,
                "h",
                AP_LONG_FLAG,
                "help",
                "Get more info ...");
            break;
        }
        fprintf(
            stderr,
            "       " _red("%s%s") "  %s%-10s" _blacki("%s\n"),
            AP_SHORT_FLAG,
            c->args[i].short_arg,
            AP_LONG_FLAG,
            c->args[i].long_arg,
            c->args[i].help);
    }
    // fprintf(stderr, "\n");
}


static inline void ap_print_parser(void) {
    fprintf(stderr, "\n %s\n ", pap->print);
    if (pap->ncommand > 1)
    {
        fprintf(stderr, _boldu("Command:\n"));
        for (size_t i = 0; i < pap->ncommand; i++)
        {
            ap_default_print_base_command(&pap->commands[i]);
        }
    }
    else
    {
        pap->print_command(NOW_CMD);
    }

    fprintf(stderr, "\n");
}


static inline void ap_print_command(void) {
    if (!pap->print_command)
    {
        ap_default_print_command(NOW_CMD);
    }
    else
    {
        pap->print_command(NOW_CMD);
    }
}

/**
 * @brief 解析命令行，内部接口
 * @param argc
 * @param argv
 */
static inline void _ap_parser_command_line(int argc, char *argv[])
{
    if (argc > 0)
    {
        if (strcmp(argv[0], "-h") == 0 || strcmp(argv[0], "--help") == 0)
        {
            ap_print_command();
            exit(0);
        }
    }
    // 当前是参数名状态，还是参数值状态
    // 为0为参数名状态
    // 为1为参数值状态
    int status = 0;

    // 解析的位置
    int count = 0;

    // 参数名的前缀长度
    int short_flag_len = strlen(AP_SHORT_FLAG);
    int long_flag_len = strlen(AP_LONG_FLAG);

    // 准备装载的参数
    ap_arg_t *arg = NULL;
    char *arg_name = NULL;
    for (;;)
    {
        if (count >= argc)
        {
            break;
        }

        // 解析器的位置
        char *need_parse = argv[count];
        int need_parse_len = strlen(need_parse);

        // 判断是否为一个参数名
        int is_short = strncmp(need_parse, AP_SHORT_FLAG, ap_min(need_parse_len, short_flag_len)) == 0;
        int is_long = strncmp(need_parse, AP_LONG_FLAG, ap_min(need_parse_len, long_flag_len)) == 0;
        int is_file = 0;

        FILE *fp = fopen(need_parse, "r");
        if(fp) is_file = 1;
        free(fp);

        if (is_short || is_long)
        {
            status = 0;
            // 去掉参数名前面的prefix
            const char *prefix;
            if (is_long)
            {
                prefix = AP_LONG_FLAG;
            }
            else
            {
                prefix = AP_SHORT_FLAG;
            }
            int prefix_len = strlen(prefix);
            arg_name = need_parse + prefix_len;

            // 检查参数是否存在
            int exist = 0;
            for (int i = 0; i < (NOW_CMD)->nargs; i++)
            {
                arg = &((NOW_CMD)->args[i]);
                int short_succ = strcmp(arg_name, arg->short_arg) == 0;
                int long_succ = strcmp(arg_name, arg->long_arg) == 0;
                if (short_succ || long_succ)
                {
                    exist = 1;
                    break;
                }
            }
            if (!exist)
            {
                fprintf(stderr, ERROR_ARG_NOT_EXIST, arg_name - prefix_len);
                exit(1);
            }
        }
        else
        {
            // 参数值/一个位置参数
            status = 1;
        }

        if (arg == NULL)
        {
            if(is_file) {
                int exist = 0;
                for (int i = 0; i < (NOW_CMD)->nargs; i++)
                {
                    arg = &((NOW_CMD)->args[i]);
                    int short_succ = strcmp("i", arg->short_arg) == 0;
                    int long_succ = strcmp("input", arg->long_arg) == 0;
                    if (short_succ || long_succ) {
                        exist = 1;
                        break;
                    }
                }
            } else {
                fprintf(stderr, ERROR_LOST_ARG_NAME, need_parse);
                exit(1);
            }
        }

        // 根据status来进行下一步的处理
        if (status && arg->arg_have_value == ap_NO)
        {
            // 不需要参数值
            fprintf(stderr, ERROR_DONOT_NEED_VALUE, arg->long_arg);
            exit(1);
        }

        if (status && arg->arg_have_value == ap_YES)
        {
            arg->value = need_parse;
            arg = NULL;
        }

        if (!status && arg->arg_have_value == ap_NO)
        {
            arg->init.b = 1;
        }
        count++;
    }
    if (arg && arg->arg_have_value == ap_YES)
    {
        fprintf(stderr, ERROR_LOST_ARG_VALUE, arg->long_arg);
        exit(1);
    }
}


void ap_do_parser(int argc, char *argv[], char *envp[]) {
    int argc_copy = argc;
    char **argv_copy = argv;
    if (argc > 1)
    {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
        {
            ap_print_parser();
            exit(0);
        }
    }

    ap.prog_name = argv[0];
    ap.prog_name_without_path = strrchr(ap.prog_name, '/');
    if (ap.prog_name_without_path)
    {
        ap.prog_name_without_path++;
    }
    else
    {
        ap.prog_name_without_path = ap.prog_name;
    }

    // 跳过文件名
    argc--;
    argv++;
    if (ap.have_subcommand)
    {
        char *subcommand;
        int exist = 0;
        if (!argc && !ap.have_global)
        { // 子命令
            ap_print_parser();
            fprintf(stderr, ERROR_NO_SUBCOMMAND);
            exit(1);
        }
        else if (argc && ap.have_global)
        {   
            // 有全局命令且有参数
            subcommand = argv[0];
            /* 在这里自动装载子命令 */
            for (int i = 0; i < pap->ncommand; i++)
            {
                if (strcmp(pap->commands[i].command, subcommand) == 0)
                {
                    pap->commands[i].subcommand = 1;
                    pap->command_pos = i;
                    exist = 1;
                    break;
                }
            }
            if (!exist)
            {
                subcommand = AP_DEFAULT_COMMAND;
            }
        }
        else if(!argc && ap.have_global) {
            // 有全局变量且无参数
            subcommand = AP_DEFAULT_COMMAND;
        }
        else
        {
            subcommand = argv[0];
            /* 在这里自动装载子命令 */
            for (int i = 0; i < pap->ncommand; i++)
            {
                if (strcmp(pap->commands[i].command, subcommand) == 0)
                {
                    pap->commands[i].subcommand = 1;
                    pap->command_pos = i;
                    exist = 1;
                    break;
                }
            }
            if (!exist)
            {
                fprintf(stderr, ERROR_SUBCOMMAND_NOT_EXIST, subcommand);
                exit(1);
            }
        }
        
    }
    else
    {
        // 只有全局命令
        pap->command_pos = 0;
    }

    /* 开始解析命令行参数 */
    _ap_parser_command_line(argc - ap.have_subcommand, argv + ap.have_subcommand);
    /* 开始调用回调函数 */
    (NOW_CMD)->callback(argc_copy, argv_copy, envp);
}
