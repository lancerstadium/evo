#include "sob.h"


// ==================================================================================== //
//                                    sob: application (APP)
// ==================================================================================== //
#ifndef SOB_APP_OFF

ArgParser_def_fn(all) {

}

ArgParser_def_fn(logs) {

}

ArgParser_def_fn(test) {

}

ArgParser_def_fn(clean) {

}


ArgParser_def_args(default_args) = {
    ArgParser_arg_INPUT,
    ArgParser_arg_OUTPUT,
    ArgParser_arg_END
};

ArgParser_def_args(test_args) = {
    { .sarg = "u", .larg = "unit", .help = "set unit test name"},
    { .sarg = "l", .larg = "list", .no_val = true, .help = "list all unit test"},
    ArgParser_arg_END
};

int main(int argc, char *argv[], char *envp[]) {

    ArgParser_init("Sob - Super Nobuild Toolkit with only .h file", NULL);
    ArgParser_use_cmd(NULL  , "run all"   , "This is usage", all    , default_args);
    ArgParser_use_cmd(NULL  , "run test"  , "This is usage", test   , test_args);
    ArgParser_use_cmd("log" , "run log"   , "This is usage", logs   , default_args);
    ArgParser_use_cmd(NULL  , "run clean" , "This is usage", clean  , default_args);
    
    ArgParser_run(argc, argv, envp);
    return 0;
}

#endif
// ==================================================================================== //
//                                    sob: application (APP)
// ==================================================================================== //