#include <evo/evo.h>
#include <evo/task.h>
#include <getopt.h>

// ==================================================================================== //
//                                    edb: Static                                     
// ==================================================================================== //

typedef struct {
    char* log_file;
    char* img_file;
    char* diff_so_file;
    int   difftest_port;
    bool  is_batch_mode;
    Task(Exec)* task;
    u64 halt_pc;
    u32 halt_ret;
} EDBGlobal;

UNUSED static EDBGlobal edb_global = {
    .log_file = NULL,
    .img_file = NULL,
    .diff_so_file = NULL,
    .difftest_port = 1234,
    .task = NULL
};

// ==================================================================================== //
//                                    edb: Img                                    
// ==================================================================================== //

static size_t EDBImg_init() {
    Val* img = Val_from_file(edb_global.img_file);
    if(!img) {
        img = Val_from_u32((u32[]){
            0x00000297,
            0x00028823,
            0x0102c503,
            0x00100073,
        }, 16);
    }
    edb_global.task = Task_init(Exec, "task-exec", img);
    return img->len;
}



// ==================================================================================== //
//                                    edb: Regex                                     
// ==================================================================================== //

void EDBRegex_init(){

}



// ==================================================================================== //
//                                    edb: WatchPoint                    
// ==================================================================================== //


void EDBWp_init() {

}

// ==================================================================================== //
//                                    edb: FrameWork    
// ==================================================================================== //


static int EDB_parse(int argc, char *argv[]) {
    const struct option table[] = {
        {"batch"    , no_argument      , NULL, 'b'},
        {"log"      , required_argument, NULL, 'l'},
        {"diff"     , required_argument, NULL, 'd'},
        {"port"     , required_argument, NULL, 'p'},
        {"help"     , no_argument      , NULL, 'h'},
        {0          , 0                , NULL,  0 },
    };
    int o;
    while ( (o = getopt_long(argc, argv, "-bhl:d:p:", table, NULL)) != -1) {
        switch (o) {
        case 'b': edb_global.is_batch_mode = true; break;
        case 'p': sscanf(optarg, "%d", &edb_global.difftest_port); break;
        case 'l': edb_global.log_file = optarg; break;
        case 'd': edb_global.diff_so_file = optarg; break;
        case  1 : edb_global.img_file = optarg; return 0;
        default:
            printf("Usage: %s [OPTION...] IMAGE [args]\n\n", argv[0]);
            printf("\t-b,--batch              run with batch mode\n");
            printf("\t-l,--log=FILE           output log to FILE\n");
            printf("\t-d,--diff=REF_SO        run DiffTest with reference REF_SO\n");
            printf("\t-p,--port=PORT          run DiffTest with port PORT\n");
            printf("\n");
            exit(0);
        }
    }
    return 0;
}

static uint64_t get_time_internal() {
#if defined(CFG_TARGET_AM)
    uint64_t us = io_read(AM_TIMER_UPTIME).us;
#elif defined(CFG_TIMER_GETTIMEOFDAY)
    struct timeval now;
    gettimeofday(&now, NULL);
    uint64_t us = now.tv_sec * 1000000 + now.tv_usec;
#else
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &now);
    uint64_t us = now.tv_sec * 1000000 + now.tv_nsec / 1000;
#endif
    return us;
}

void EDBRand_init() {
  srand(get_time_internal());
}

static void EDB_welcome() {
    Log_info("Build time: %s, %s", __TIME__, __DATE__);
    printf("Welcome to EDB %s!\n", _BLUE(STR(CFG_SISA)));
    printf("For help, type \"help\"\n");
}

void EDB_init(int argc, char *argv[]) {
    EDB_parse(argc, argv);
    EDBRand_init();
    EDBRegex_init();
    EDBImg_init();
    EDBWp_init();
    EDB_welcome();
}

// ==================================================================================== //
//                                    edb: Main Loop                                     
// ==================================================================================== //

#define CPU0(t)  ((CPUState(ISE) *)((t)->ctx.cpu))
#define NR_CMD  ARRLEN(cmd_table)

static int cmd_q(char* args);
static int cmd_c(char* args);
static int cmd_si(char* args);
static int cmd_info(char* args);
static int cmd_p(char* args);
static int cmd_tsp(char* args);
static int cmd_x(char* args);
static int cmd_w(char* args);
static int cmd_d(char* args);
static int cmd_help(char *args);

static struct {
    const char *name;
    const char *description;
    int (*handler) (char *);
} cmd_table [] = {
    { "help", "Display information about all supported commands"  , cmd_help },
    { "info", "Info of [r/w]"                                     , cmd_info },
    { "si"  , "Single step execution [N]"                         , cmd_si   },
    { "p"   , "Caculate the value of [expr]"                      , cmd_p    },
    { "tsp" , "Test exprs in [file]"                              , cmd_tsp  },
    { "x"   , "Scan Memory [N expr]"                              , cmd_x    },
    { "w"   , "Set watchpoint [expr]"                             , cmd_w    },
    { "d"   , "Delete watchpoint [N]"                             , cmd_d    },
    { "c"   , "Continue the execution of the program"             , cmd_c    },
    { "q"   , "Exit NEMU"                                         , cmd_q    },
    /* TODO: Add more commands */
};

static int cmd_c(UNUSED char *args) {
    Task_run(Exec, edb_global.task);
    return 0;
}

static int cmd_si(char *args) {
    char *sub = strtok(args, " ");
    uint64_t n = 1;
    if (sub != NULL) {
        n = strtol(sub, NULL, 10);
    }
    Task_rundbg(Exec, edb_global.task, Val_new_u64(n));
    return 0;
}

static int cmd_info(char *args) {
    char *sub = strtok(args, " ");
    if (sub == NULL) {
        printf("Usage: info [r|w]\n");
    } else {
        if (strcmp(sub, "r") == 0) {
            // isa_reg_display();
        } else if (strcmp(sub, "w") == 0) {
            // info_wp(-1);
        } else {
            printf("Usage: info [r|w]\n");
        }
    }
    return 0;
}

static int cmd_x(char *args) {
    char *sub = strtok(args, " ");
    if (sub == NULL) {
        printf("Usage: x [N] [expr]\n");
    } else {
        /// TODO: implement x
    }
    return 0;
}

static int cmd_p(char *args) {
    char *sub = strtok(args, " ");
    if (sub == NULL) {
        printf("Usage: p [expr]\n");
    } else {
        // bool success;
        // unsigned res = expr(sub, &success);
        // if (success) printf("%s = %u\n", sub, res);
    }
    return 0;
}
static int cmd_tsp(char *args) {
    char *sub = strtok(args, " ");
    if (sub == NULL) {
        printf("Usage: tsp [file]\n");
    } else {
        // 1. 读入文件
        FILE *fp = fopen(sub, "r");
        if (fp == NULL) {
            printf("Open file %s failed.\n", sub);
            return 0;
        }
        // // 2. 读取每一行格式：`ref, exprisson` `int char*`
        // char line[1024];
        // while (fgets(line, 1024, fp) != NULL) {
        //     char *p = line;
        //     unsigned ref = atoi(strtok(p, " "));
        //     char *exprisson = strtok(NULL, " ");
        //     // 删除换行符
        //     if (exprisson[strlen(exprisson) - 1] == '\n') {
        //         exprisson[strlen(exprisson) - 1] = '\0';
        //     }

        //     bool success;
        //     unsigned res = expr(exprisson, &success);
        //     if (success && res == ref) {
        //         printf("\033[0;32msucc:\033[0m %u == %u = %s\n",  res, ref, exprisson);
        //     } else if (success && res != ref) {
        //         printf("\033[0;31mfail:\033[0m %u != %u = %s\n",  res, ref, exprisson);
        //     }
        // }
    }
    return 0;
}

static int cmd_w(char *args) {
    char *sub = strtok(args, " ");
    if (sub == NULL) {
        printf("Usage: w [expr]\n");
    } else {
        // new_wp(sub);
    }
    return 0;
}

static int cmd_d(char *args) {
    char *sub = strtok(args, " ");
    if (sub != NULL) {
        int n = atoi(sub);
        // free_wp(n);
    } else {
        printf("Usage: d [n]\n");
    }
    return 0;
}

static int cmd_q(UNUSED char* args) {
    TaskCtx_OP(Exec, set_status)(&edb_global.task->ctx, CPU_QUIT);
    return -1;
}

static int cmd_help(UNUSED char *args){
    /* extract the first argument */
    char *arg = strtok(NULL, " ");
    size_t i;
    if (arg == NULL) {
        /* no argument given */
        for (i = 0; i < NR_CMD; i ++) {
            printf("\t%s - %s\n", cmd_table[i].name, cmd_table[i].description);
        }
    } else {
        for (i = 0; i < NR_CMD; i ++) {
            if (strcmp(arg, cmd_table[i].name) == 0) {
                printf("\t%s - %s\n", cmd_table[i].name, cmd_table[i].description);
                return 0;
            }
        }
        Log_warn("Unknown command '%s'\n", arg);
    }
    return 0;
}

void EDB_loop() {
    char* line;
    while((line = linenoise("(EDB) ")) != NULL) {
        char* line_end = line + strlen(line);
        /* extract the first token as the command */
        char *cmd = strtok(line, " ");
        if (cmd == NULL) {
            linenoiseFree(line);
            continue;
        }
        /* treat the remaining string as the arguments,
         * which may need further parsing */
        char *args = cmd + strlen(cmd) + 1;
        if (args >= line_end) {
            args = NULL;
        }
        size_t i;
        for (i = 0; i < NR_CMD; i ++) {
            if (strcmp(cmd, cmd_table[i].name) == 0) {
                if (cmd_table[i].handler(args) < 0) {
                    linenoiseFree(line);
                    return;
                }
                break;
            }
        }
        if (i == NR_CMD) {
            Log_warn("Unknown command '%s'\n", cmd);
        }
        linenoiseFree(line);
    }
}

#undef CPU0
#undef NR_CMD

// ==================================================================================== //
//                                    edb: Prog Entry                                      
// ==================================================================================== //

int main(int argc, char *argv[]) {
    EDB_init(argc, argv);
    EDB_loop();
    return 0;
}