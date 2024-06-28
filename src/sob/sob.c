#include "sob.h"


typedef struct {
    char n[32];     // name
    size_t w;       // bit width
    bool e;         // false: little endian, true: big endian
} Arch;

#define Arch_width(A)       ((A).w)
#define Arch_name(A)        ((A).n)
#define Arch_endian(A)      ((A).e)

typedef enum {
    UNKNOWN_ARCH,
    I386_ARCH,
    X86_64_ARCH,
    ARM_ARCH,
    AARCH64_ARCH,
    RISCV32_ARCH,
    ARCHID_SIZE
} ArchID;


UNUSED static Arch arch_map[ARCHID_SIZE] = {
    [UNKNOWN_ARCH]  = {"unknown",  0, false},
    [I386_ARCH]     = {"i386"   , 32, false},
    [X86_64_ARCH]   = {"x86_64" , 64, false},
    [RISCV32_ARCH]  = {"riscv32", 32, false}
};


// ==================================================================================== //
//                                    sob: application (APP)
// ==================================================================================== //
#ifndef SOB_APP_OFF
#define VALGRIND

CStrArray_def()


ArgParser_def_fn(all) {
    ECHO("Hello World\n");
}

ArgParser_def_fn(logs) {
    int n;
    EXES(n, "tail -n $(($(tac test/tests.log | grep -m 1 -n '^───── Run' | cut -d: -f1) + 1)) test/tests.log | sed '/^$/d'");
}

ArgParser_def_fn(sys) {
    CStr* cmd1, *cmd2, *cmd3;
    CStrArray_init(&cmd1, "uname", "-a", NULL);
    CStrArray_init(&cmd3, "isam", "wuhu", NULL);
    CStrArray_display(cmd1);
    CStrArray_from(&cmd2, "ls -l");
    CStrArray_display(cmd3);
    CStrArray_set(cmd3, 0, "sll");
    CStrArray_display(cmd3);

    EXEC("ls -l");
    EXEC("echo nihao");
}

ArgParser_def_fn(test) {
    FILE *fp;
    char buffer[1024];
    char cmd_buf[1024+128];

    ArgParser_get("unit");

    if(ArgParser_cur_arg && ArgParser_cur_arg->literal) {
        ECHO(_WHITE_BD_UL("Running Unit Tests:") "\n");
        ArgParserArg* test_arg = ArgParser_cur_arg;
        Str_trim(test_arg->literal);
        char test_buf[32] = "./test/";
        strcat(test_buf, test_arg->literal);
        if(IS_FILE(strcat(test_buf, "_tests"))) {
            STR_FMT(cmd_buf, "%s 2>> test/tests.log", test_buf);
            int n;
            EXES(n, cmd_buf);
            if(n == 0) {
                ECHO(_GREEN("[Test %s Passed]") "\n", test_arg->literal);
            } else {
                ECHO(_RED("[Test %s Failed]") "\n", test_arg->literal);
                ECHO(_CYAN("[LOG]") "\n");
                logs(argc, argv, envp);
            }
        } else {
            ECHO(_RED("[Test %s not Found]") "\n", test_arg->literal);
        }
    } else {
        EXEF(fp, "r", "find test -type f -name '*_tests'");
        ArgParser_get("list");
        if(ArgParser_cur_arg && ArgParser_cur_arg->init.b) {
            ECHO(_WHITE_BD_UL("Unit Tests List:") "\n");
            for (size_t i = 0; fgets(buffer, sizeof(buffer), fp) != NULL; i++) {
                Str_trim(buffer);
                if(IS_FILE(buffer)) {
                    ECHO(_MAGENTA("%2lu") ": " _BLUE("%s") "\n", i, buffer);
                }
            }
        } else {
            ECHO(_WHITE_BD_UL("Running Unit Tests:") "\n");
            for (size_t i = 0; fgets(buffer, sizeof(buffer), fp) != NULL; i++) {
                Str_trim(buffer);
                if(IS_FILE(buffer)) {
                    STR_FMTN(cmd_buf, sizeof(cmd_buf), "./%s 2>> test/tests.log", buffer);
                    // STR_FMT(cmd_buf, "./%s 2>> test/tests.log | sed 's/\x1B\[[0-9;]*[JKmsu]//g' > test/tests.report", buffer);
                    int n;
                    EXES(n, cmd_buf);
                    if(n == 0) {
                        ECHO(_GREEN("[Test %lu Passed]") "\n", i);
                    } else {
                        ECHO(_RED("[Test %lu Failed]") "\n", i);
                        ECHO(_CYAN("[LOG]") "\n");
                        logs(argc, argv, envp);
                        break;
                    }
                } else {
                    ECHO(_RED("[Test %s not Found]") "\n", buffer);
                    break;
                }
            }
        }
        EXNF(fp);
    }
    
}

ArgParser_def_fn(clean) {
    int n;
    EXEC("rm -rf build");
    EXEC("rm -f test/tests.log test/valgrind.log");
    EXES(n, "rm -rf test/*_tests");
    EXES(n, "rm -rf *.out *.elf");
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
    ArgParser_use_cmd(NULL  , "run sys"   , "This is usage", sys    , default_args);
    ArgParser_use_cmd(NULL  , "run test"  , "This is usage", test   , test_args);
    ArgParser_use_cmd("log" , "run log"   , "This is usage", logs   , default_args);
    ArgParser_use_cmd(NULL  , "run clean" , "This is usage", clean  , default_args);
    
    ArgParser_sys_cmd("uname -a");
    ArgParser_sys_cmd("perf record -e cycles -F 999 ls -l");

    ArgParser_run(argc, argv, envp);
    return 0;
}

#endif
// ==================================================================================== //
//                                    sob: application (APP)
// ==================================================================================== //