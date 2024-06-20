#include "sob.h"



#ifndef SOB_APP_OFF


CStrArray_def()

ArgParser_def_fn(all) {
    printf("Hello World\n");
}

ArgParser_def_fn(sys) {
    CStr* cmd1, *cmd2, *cmd3;
    CStr* cmd4, *cmd5;
    CStrArray_init(&cmd1, "uname", "-a", NULL);
    CStrArray_init(&cmd3, "isam", NULL);
    CStrArray_display(cmd1);

    CStrArray_from(&cmd2, "ls -l");
    
    CStrArray_display(cmd3);

    // CStrArray_pushn(cmd2, "AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH");
    // CStrArray_new(cmd3, "make", "all");
    // CStrArray_from(cmd4, "ls -l -a");
    // CStrArray_from(cmd5, "echo db");
    
    // CStrArray_display(cmd4);
    // CStrArray_display(cmd5);

    /// TODO: 注释一个可执行
    EXEC("ls -l");
    EXEC("echo nihao");


    // CStr mm;
    // CStrArray_pop(cmd2, mm);
    // printf("mm: %s\n", mm);

    // bool b = IS_DIR("./doc");
    // printf("%s\n", b ? "true" : "false");


    // CStr* files;
    // LIST_FILES("./", files);
    // CStrArray_forauto(files, i, file, {
    //     printf("- %s\n", file);
    // });

    // Sob_rename(mm, "hello");

    /// TODO: CStrArray_copy memory leak
    // CStr* cmd2_copy;
    // CStrArray_copy(cmd2, cmd2_copy);
    // CStrArray_prefix(cmd2_copy, "-L");
    // CStrArray_forauto(cmd2, i, s,
    //     printf("copy: %s\n", s);
    // );

    /// TODO: CStrArray_join memory leak
    // char* cc;
    // CStrArray_path(cmd2, cc);
    // printf("%s\n", cc);
    // CStrArray_join(cmd2, cc, ",");
    // printf("%s\n", cc);

    /// TODO: Segment fault maybe stack overflow
    // MKDIR("demo01", "demo02", "demo03");
    // RM("demo01", "demo02", "demo03");

    // TOUCH("demo01.txt", "demo02.txt", "demo03.txt");
    // RM("demo01.txt", "demo02.txt", "demo03.txt");

    // printf("%s\n", STR_BOOL(IS_MODIFIED_AFTER("./demo01", "./demo02")));
}

ArgParser_def_fn(test) {
    printf(_WHITE_BD_UL("Running Unit Tests:") "\n");
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
    

    ArgParser_sys_cmd("uname -a");
    ArgParser_sys_cmd("ulimit -s");
    ArgParser_sys_cmd("perf record -e cycles -F 999 ls -l");
    ArgParser_sys_cmd("echo nihao");

    ArgParser_run(argc, argv, envp);
    return 0;
}

#endif