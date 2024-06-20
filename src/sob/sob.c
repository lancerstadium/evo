#include "sob.h"



#ifndef SOB_APP_OFF


#define VALGRIND

CStrArray_def()

ArgParser_def_fn(all) {
    ECHO("Hello World\n");
}

ArgParser_def_fn(log) {
    int n;
    EXES(n, "tail -n $(($(tac test/tests.log | grep -m 1 -n '^───── Run' | cut -d: -f1) + 1)) test/tests.log | sed '/^$/d'");
}

ArgParser_def_fn(sys) {
    CStr* cmd1, *cmd2, *cmd3;
    CStrArray_init(&cmd1, "uname", "-a", NULL);
    CStrArray_init(&cmd3, "isam", NULL);
    CStrArray_display(cmd1);
    CStrArray_from(&cmd2, "ls -l");
    CStrArray_display(cmd3);

    EXEC("ls -l");
    EXEC("echo nihao");
}

ArgParser_def_fn(test) {
    ECHO(_WHITE_BD_UL("Running Unit Tests:") "\n");
    FILE *fp;
    char buffer[1024];
    char cmd_buf[1024];

    EXEF(fp, "r", "find test -type f -name '*_tests'");
    for (size_t i = 0; fgets(buffer, sizeof(buffer), fp) != NULL; i++) {
        Str_trim(buffer);
        if(IS_FILE(buffer)) {
            STR_FMT(cmd_buf, "./%s 2>> test/tests.log", buffer);
            // STR_FMT(cmd_buf, "./%s 2>> test/tests.log | sed 's/\x1B\[[0-9;]*[JKmsu]//g' > test/tests.report", buffer);
            int n;
            EXES(n, cmd_buf);
            if(n == 0) {
                ECHO(_GREEN("[Test %lu Passed]") "\n", i);
            } else {
                ECHO(_RED("[Test %lu Failed]") "\n", i);
                ECHO(_CYAN("[LOG]") "\n");
                log(argc, argv, envp);
                break;
            }
        } else {
            ECHO(_RED("[Test %lu not Found]") "\n", i);
            break;
        }
    }
    EXNF(fp);
}


ArgParser_def_args(default_args) = {
    ArgParser_arg_INPUT,
    ArgParser_arg_OUTPUT,
    ArgParser_arg_END
};

int main(int argc, char *argv[], char *envp[]) {

    ArgParser_init("Sob - Super Nobuild Toolkit with only .h file", NULL);
    ArgParser_use_cmd(NULL, "run all" , "This is usage", all  , default_args);
    ArgParser_use_cmd(NULL, "run sys" , "This is usage", sys, default_args);
    ArgParser_use_cmd(NULL, "run test", "This is usage", test , default_args);
    ArgParser_use_cmd(NULL, "run log" , "This is usage", log , default_args);
    
    ArgParser_sys_cmd("uname -a");
    ArgParser_sys_cmd("perf record -e cycles -F 999 ls -l");

    ArgParser_run(argc, argv, envp);
    return 0;
}

#endif