

// ==================================================================================== //
//                                       Include
// ==================================================================================== //

#include "evo.h"

// ==================================================================================== //
//                                Commands & Args Define
// ==================================================================================== //

// Step1: 定义参数
ap_def_args(default_args) = {
    {.short_arg = "o", .long_arg = "output", .init.s = "./a.out", .help = "set output path"},
    {.short_arg = "q", .long_arg = "quiet",  .init.i = 3, .help = "set quiet level"},
    AP_INPUT_ARG,
    AP_END_ARG};

ap_def_args(test_args) = {
    {.short_arg = "o", .long_arg = "output", .init.s = "./test", .help = "set output path"},
    {.short_arg = "q", .long_arg = "quiet",  .init.i = 3, .help = "set quiet level"},
    AP_END_ARG};

void arg_parser(int argc, char *argv[], char *envp[]) {
    // 初始化解析器
    ap_init_parser("evo - Evolvable Programming Language", NULL);
    // 添加命令
    ap_add_command("default"    , "evo compiler"            , "This is usage."  , evoc              , default_args);
    ap_add_command("hello"      , "Print `Hello, World!`."  , "This is usage."  , hello_world       , default_args);
    // ap_add_command("test"       , "Unit test"               , "This is usage."  , util_test         , test_args);
    // 开始解析
    ap_do_parser(argc, argv, envp);
}

// ==================================================================================== //
//                                  Proc Entry: evo
// ==================================================================================== //

int main(int argc, char *argv[], char *envp[]) {
    arg_parser(argc, argv, envp);
    return 0;
}