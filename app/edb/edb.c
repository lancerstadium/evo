#include <evo/evo.h>
#include <evo/task.h>
#include <getopt.h>
#include <regex.h>


// ==================================================================================== //
//                                    edb: Defined                                    
// ==================================================================================== //


#define CPU0(t)  ((CPUState(ISE) *)((t)->ctx.cpu))
#define NR_CMD  ARRLEN(cmd_table)
#define NR_REGEX ARRLEN(rules)
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
//                                    edb: ISA                                    
// ==================================================================================== //


void isa_reg_display() {
//   for (int i = 0; i < MUXDEF(CONFIG_RVE, 16, 32); ++i) {
//     printf("%3s: 0x%x\t", regs[i], cpu.gpr[i]);
//     if(i % 4 == 3) {
//       printf("\n");
//     }
//   }
}

u64 isa_reg_str2val(const char *s, bool *success) {
//   for (int i = 0; i < MUXDEF(CONFIG_RVE, 16, 32); ++i) {
//     if (strcmp(s, regs[i]) == 0) {
//       *success = true;
//       return cpu.gpr[i];
//     }
//   }
//   if (strcmp(s, "pc") == 0) {
//     *success = true;
//     return cpu.pc;
//   }
  *success = false;
  return 0;
}

// ==================================================================================== //
//                                    edb: Regex                                     
// ==================================================================================== //

enum {
    TK_NOTYPE = 256, 
    TK_EQ, TK_NE,
    TK_AND,
    TK_DEC,
    TK_HEX,
    TK_REG,
    /* TODO: Add more token types */
};

typedef struct {
    const char *regex;
    int token_type;
} Rule;

UNUSED static Rule rules[] = {
    {" +"           , TK_NOTYPE },      // spaces
    {"\\+"          , '+'       },      // plus
    {"\\-"          , '-'       },      // minus
    {"\\*"          , '*'       },      // asterisk
    {"\\/"          , '/'       },      // slash
    {"\\("          , '('       },      // lparen
    {"\\)"          , ')'       },      // rparen
    
    {"=="           , TK_EQ     },      // equal
    {"!="           , TK_NE     },      // not equal
    {"&&"           , TK_AND    },      // and

    {"\\$[0-9a-z]+" , TK_REG    },      // register
    {"0x[0-9a-f]+"  , TK_HEX    },      // hex
    {"[0-9]+(u)?"   , TK_DEC    },      // decimal
};

static regex_t re[NR_REGEX] = {};

/* Rules are used for many times.
 * Therefore we compile them only once before any usage. */
void EDBRegex_init(){
    char error_msg[128];
    int ret;
    for (size_t i = 0; i < NR_REGEX; i ++) {
        ret = regcomp(&re[i], rules[i].regex, REG_EXTENDED);
        if (ret != 0) {
            regerror(ret, &re[i], error_msg, 128);
            Log_err("regex compilation failed: %s\n%s", error_msg, rules[i].regex);
            exit(1);
        }
    }
}

typedef struct token {
    int type;
    char str[32];
} Token;

#define TOKEN_MAX_LEN 1024
static Token tokens[TOKEN_MAX_LEN] __attribute__((used)) = {};
static int nr_token __attribute__((used))  = 0;

static bool Token_make(char *e) {
    int position = 0;
    int i;
    regmatch_t pmatch;
    nr_token = 0;

    while (e[position] != '\0') {
        /* Try all rules one by one. */
        for (i = 0; i < NR_REGEX; i ++) {
            if (regexec(&re[i], e + position, 1, &pmatch, 0) == 0 && pmatch.rm_so == 0) {
                char *substr_start = e + position;
                int substr_len = pmatch.rm_eo;
                position += substr_len;

                /* TODO: Now a new token is recognized with rules[i]. Add codes
                * to record the token in the array `tokens'. For certain types
                * of tokens, some extra actions should be performed.
                */

                Token *t = &tokens[nr_token ++];
                t->type = rules[i].token_type;
                strncpy(t->str, substr_start, substr_len);

                switch (rules[i].token_type) {
                    case '+': break;
                    case '-': break;
                    case '*': break;
                    case '/': break;
                    case '(': break;
                    case ')': break;
                    case TK_EQ: break;
                    case TK_NE: break;
                    case TK_AND: break;
                    case TK_DEC: break;
                    case TK_HEX: break;
                    case TK_REG: break;
                    case TK_NOTYPE: break;
                    default: TODO();
                }
                break;
            }
        }
        if (i == NR_REGEX) {
            Log_warn("no match at position %d\n%s\n%*.s^\n", position, e, position, "");
            return false;
        }
    }
    return true;
}

static void Token_flush() {
    nr_token = 0;
    for (int i = 0; i < TOKEN_MAX_LEN; i ++) {
        tokens[i].type = TK_NOTYPE;
        tokens[i].str[0] = '\0';
    }
}

static int find_op(int p, int q, bool* success) {
    int depth = 0;
    int res = p;
    int res_level = 10;
    *success = false;
    for (int i = p; i <= q; i++) {
        switch (tokens[i].type) {
            case '(':
                depth++;
                break;
            case ')':
                depth--;
                break;
            case TK_EQ:
            case TK_NE:
            case TK_AND:
                if (depth == 0 && res_level > 1) {
                    *success = true;
                    res = i;
                    res_level = 1;
                }
                break;
            case '+':
            case '-':
                if (depth == 0 && res_level > 2) {
                    *success = true;
                    res = i;
                    res_level = 2;
                }
                break;
            case '/':
                if (depth == 0 && res_level > 3) {
                    *success = true;
                    res = i;
                    res_level = 3;
                }
            case '*':
                if (depth == 0 && res_level > 4) {
                    *success = true;
                    res = i;
                    res_level = 4;
                }
                break;
        }
    }
    return res;
}

static u64 eval(int p, int q) {
    char* e_p = tokens[p].str;
    int t_p = tokens[p].type;
    char* e_q = tokens[q].str;
    int t_q = tokens[q].type;
    if (p > q) {
        printf("Bad expr range: [%d, %d]\n", p, q);
        return 0;
    } else if (p == q) {
        u64 res = 0;
        if (t_p == TK_DEC)      { res = atoi(e_p);              }
        else if (t_p == TK_HEX) { res = strtol(e_p, NULL, 16);  }
        else if (t_p == TK_REG) {
            bool success;
            res = isa_reg_str2val(e_p + 1, &success);
            if (!success) { printf("Bad register: %s\n", e_p);  }
        }
        return res;
    } else if (t_p == '(' && t_q == ')') {
        return eval(p + 1, q - 1);
    } else if ((p < q) && (t_p == TK_NOTYPE)) {
        return eval(p + 1, q);
    } else if ((p < q) && (t_q == TK_NOTYPE)) {
        return eval(p, q - 1);
    } else {
        bool success;
        int op = find_op(p, q, &success);
        if (!success) {
            printf("Bad expr: (s: %s, e: %s)\n", e_p, e_q);
            return 0;
        }
        if (op == p) {  // Unary: +a, -a, *a
            u64 a = eval(op + 1, q);
            switch (tokens[op].type) {
                case '+': return a;
                case '-': return -a;
                case '*': return a;
                default: TODO();
            }
        } else {        // Binary: 
            u64 a = eval(p, op - 1);
            u64 b = eval(op + 1, q);
            switch (tokens[op].type) {
                case '+': return (unsigned)a + (unsigned)b;
                case '-': return (unsigned)a - (unsigned)b;
                case '*': return (unsigned)a * (unsigned)b;
                case '/': return (unsigned)a / (unsigned)b;
                case TK_EQ: return a == b;
                case TK_NE: return a != b;
                case TK_AND: return a && b;
                default: TODO();
            }
        }
    }
    printf("Expr eval fail: (s: %s, e: %s)\n", e_p, e_q);
    return 0;
}


u64 expr(char *e, bool *success) {
    if (!Token_make(e)) {
        *success = false;
        return 0;
    }
    /* TODO: Insert codes to evaluate the expression. */

    *success = true;
    int p = 0, q = nr_token - 1;

    u64 res = eval(p, q);
    Token_flush();
    return res;
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
    { "q"   , "Exit EDB"                                          , cmd_q    },
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
#undef NR_REGEX
#undef NR_CMD

// ==================================================================================== //
//                                    edb: Prog Entry                                      
// ==================================================================================== //

int main(int argc, char *argv[]) {
    EDB_init(argc, argv);
    EDB_loop();
    return 0;
}