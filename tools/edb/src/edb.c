#include "sob.h"
#include "linenoise.h"
#include <evo.h>
#include <getopt.h>
#include <regex.h>

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// ==================================================================================== //
//                                    edb: Defined                                    
// ==================================================================================== //

#define EDB_MODLE           onnx
#define EDB_DEVICE          cpu
#define EDB_NCMD            ARRLEN(cmd_table)
#define EDB_NWP             32
#define EDB_NREGEX          ARRLEN(rules)
#define EDB_MAX_TOKEN_LEN   1024
#define EDB_DIFF_POST       0xd1ff
#define EDB_BUFFER_SIZE     1024
#ifndef  EDB_DIFF_SERVER
#define EDB_INFO(...)       printf(__VA_ARGS__)
#else
#define EDB_INFO(...)                                                 \
    do {                                                              \
        memset(edb.diff_buf, 0, strlen(edb.diff_buf) * sizeof(char)); \
        sprintf(edb.diff_buf, __VA_ARGS__);                           \
        edb.diff_buf[strlen(edb.diff_buf)] = ' ';                     \
        send(edb.connect, edb.diff_buf, strlen(edb.diff_buf), 0);     \
    } while (0)

#endif

// ==================================================================================== //
//                                    edb: Static                                     
// ==================================================================================== //

// EDB
typedef struct {
    char* log_file;
    char* model_file;
    char* diff_file;
    char  diff_buf[EDB_BUFFER_SIZE];
    int   diff_port;
    int   listen;
    int   connect;
    bool  is_batch_mode;
    // evo var
    serializer_t * sez;
    context_t * ctx;
    tensor_t * in;
    tensor_t * out;
} edb_t;

static edb_t edb = {
    .log_file = NULL,
    .model_file = NULL,
    .diff_file = NULL,
    .diff_port = EDB_DIFF_POST,
    .listen = 0,
    .connect = 0,
    .sez = NULL,
    .ctx = NULL,
    .in = NULL,
    .out = NULL,
};

// Regex
enum {
    EDB_TK_NOTYPE = 256, 
    EDB_TK_EQ, EDB_TK_NE,
    EDB_TK_AND,
    EDB_TK_DEC,
    EDB_TK_HEX,
    EDB_TK_REG,
    /* TODO: Add more token types */
};

typedef struct {
    const char *regex;
    int token_type;
} edb_rule_t;

static edb_rule_t rules[] = {
    {" +"           , EDB_TK_NOTYPE },      // spaces
    {"\\+"          , '+'       },      // plus
    {"\\-"          , '-'       },      // minus
    {"\\*"          , '*'       },      // asterisk
    {"\\/"          , '/'       },      // slash
    {"\\("          , '('       },      // lparen
    {"\\)"          , ')'       },      // rparen
    
    {"=="           , EDB_TK_EQ     },      // equal
    {"!="           , EDB_TK_NE     },      // not equal
    {"&&"           , EDB_TK_AND    },      // and

    {"\\$[0-9a-z]+" , EDB_TK_REG    },      // register
    {"0x[0-9a-f]+"  , EDB_TK_HEX    },      // hex
    {"[0-9]+(u)?"   , EDB_TK_DEC    },      // decimal
};

static regex_t re[EDB_NREGEX] = {};

typedef struct {
    int type;
    char str[32];
} edb_token_t;

static edb_token_t tokens[EDB_MAX_TOKEN_LEN] __attribute__((used)) = {};
static int ntoken __attribute__((used))  = 0;

// Watch Point
typedef struct edb_wp_t {
    int NO;
    struct edb_wp_t *next;

    /* TODO: Add more members if necessary */
    char* e;
    uint64_t res;
} edb_wp_t;

static edb_wp_t wp_pool[EDB_NWP] = {};
static edb_wp_t *head = NULL, *free_ = NULL;


// ==================================================================================== //
//                                    edb: Model                                
// ==================================================================================== //

static size_t edb_model_init() {
    device_reg(STR(EDB_DEVICE));
    edb.sez = serializer_new("onnx");
    if(!edb.model_file) {
        edb.model_file = "./tests/model/mnist_8/model.onnx";
        edb.in = edb.sez->load_tensor("./tests/model/mnist_8/test_data_set_0/input_0.pb");
    }
    edb.ctx = edb.sez->load_model(edb.sez, edb.model_file);
    if(edb.ctx && edb.ctx->model_size > 0) {
        printf("Load model: %s(%u Byte) success!\n", edb.model_file, edb.ctx->model_size);
        if(edb.ctx->graph) {
            graph_prerun(edb.ctx->graph);
            printf("Graph Pre-run success!\n");
        }else{
            printf("Graph Pre-run fail!\n");
        }
    } else {
        printf("Load model: %s fail!\n", edb.model_file);
    }
    return 0;
}

// ==================================================================================== //
//                                    edb: Display                                   
// ==================================================================================== //

void graph_display() {
    graph_dump(edb.ctx->graph);
}

uint64_t tensor_display(char *s, bool *success) {
    tensor_t * ts = context_get_tensor(edb.ctx, s);
    tensor_dump(ts);
    *success = true;
    return 0;
}

// ==================================================================================== //
//                                    edb: Regex                                     
// ==================================================================================== //

/* Rules are used for many times.
 * Therefore we compile them only once before any usage. */
void edb_regex_init(){
    char error_msg[128];
    int ret;
    for (size_t i = 0; i < EDB_NREGEX; i ++) {
        ret = regcomp(&re[i], rules[i].regex, REG_EXTENDED);
        if (ret != 0) {
            regerror(ret, &re[i], error_msg, 128);
            Log_err("regex compilation failed: %s\n%s", error_msg, rules[i].regex);
            exit(1);
        }
    }
}

static bool token_make(char *e) {
    int position = 0;
    size_t i;
    regmatch_t pmatch;
    ntoken = 0;

    while (e[position] != '\0') {
        /* Try all rules one by one. */
        for (i = 0; i < EDB_NREGEX; i ++) {
            if (regexec(&re[i], e + position, 1, &pmatch, 0) == 0 && pmatch.rm_so == 0) {
                char *substr_start = e + position;
                int substr_len = pmatch.rm_eo;
                position += substr_len;

                /* TODO: Now a new token is recognized with rules[i]. Add codes
                * to record the token in the array `tokens'. For certain types
                * of tokens, some extra actions should be performed.
                */

                edb_token_t *t = &tokens[ntoken ++];
                t->type = rules[i].token_type;
                strncpy(t->str, substr_start, substr_len);

                switch (rules[i].token_type) {
                    case '+': break;
                    case '-': break;
                    case '*': break;
                    case '/': break;
                    case '(': break;
                    case ')': break;
                    case EDB_TK_EQ: break;
                    case EDB_TK_NE: break;
                    case EDB_TK_AND: break;
                    case EDB_TK_DEC: break;
                    case EDB_TK_HEX: break;
                    case EDB_TK_REG: break;
                    case EDB_TK_NOTYPE: break;
                    default: TODO();
                }
                break;
            }
        }
        if (i == EDB_NREGEX) {
            Log_warn("no match at position %d\n%s\n%*.s^", position, e, position, "");
            return false;
        }
    }
    return true;
}

static void token_flush() {
    ntoken = 0;
    for (int i = 0; i < EDB_MAX_TOKEN_LEN; i ++) {
        tokens[i].type = EDB_TK_NOTYPE;
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
            case EDB_TK_EQ:
            case EDB_TK_NE:
            case EDB_TK_AND:
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
                break;
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

static uint64_t eval(int p, int q) {
    char* e_p = tokens[p].str;
    int t_p = tokens[p].type;
    char* e_q = tokens[q].str;
    int t_q = tokens[q].type;
    if (p > q) {
        EDB_INFO("Bad expr range: [%d, %d]\n", p, q);
        return 0;
    } else if (p == q) {
        uint64_t res = 0;
        if (t_p == EDB_TK_DEC)      { res = atoi(e_p);              }
        else if (t_p == EDB_TK_HEX) { res = strtol(e_p, NULL, 16);  }
        else if (t_p == EDB_TK_REG) {
            bool success;
            res = tensor_display(e_p + 1, &success);
            if (!success) { EDB_INFO("Bad register: %s\n", e_p);  }
        }
        return res;
    } else if (t_p == '(' && t_q == ')') {
        return eval(p + 1, q - 1);
    } else if ((p < q) && (t_p == EDB_TK_NOTYPE)) {
        return eval(p + 1, q);
    } else if ((p < q) && (t_q == EDB_TK_NOTYPE)) {
        return eval(p, q - 1);
    } else {
        bool success;
        int op = find_op(p, q, &success);
        if (!success) {
            EDB_INFO("Bad expr: (s: %s, e: %s)\n", e_p, e_q);
            return 0;
        }
        if (op == p) {  // Unary: +a, -a, *a
            uint64_t a = eval(op + 1, q);
            switch (tokens[op].type) {
                case '+': return a;
                case '-': return -a;
                case '*': return a;
                default: TODO();
            }
        } else {        // Binary: 
            uint64_t a = eval(p, op - 1);
            uint64_t b = eval(op + 1, q);
            switch (tokens[op].type) {
                case '+': return (unsigned)a + (unsigned)b;
                case '-': return (unsigned)a - (unsigned)b;
                case '*': return (unsigned)a * (unsigned)b;
                case '/': return (unsigned)a / (unsigned)b;
                case EDB_TK_EQ: return a == b;
                case EDB_TK_NE: return a != b;
                case EDB_TK_AND: return a && b;
                default: TODO();
            }
        }
    }
    EDB_INFO("Expr eval fail: (s: %s, e: %s)\n", e_p, e_q);
    return 0;
}


uint64_t expr(char *e, bool *success) {
    if (!token_make(e)) {
        *success = false;
        return 0;
    }
    /* TODO: Insert codes to evaluate the expression. */

    *success = true;
    int p = 0, q = ntoken - 1;

    uint64_t res = eval(p, q);
    token_flush();
    return res;
}


// ==================================================================================== //
//                                    edb: WatchPoint                    
// ==================================================================================== //


void edb_wp_pool_init() {
    int i;
    for (i = 0; i < EDB_NWP; i ++) {
        wp_pool[i].NO = i;
        wp_pool[i].next = (i == EDB_NWP - 1 ? NULL : &wp_pool[i + 1]);
    }
    head = NULL;
    free_ = wp_pool;
}

int edb_wp_new(char* e) {
    if (free_ == NULL) {
        Log_warn("No extra free wp\n");
        return -1;
    }
    edb_wp_t* wp = free_;
    free_ = free_->next;

    // deal with expr
    bool success;
    uint64_t res = expr(e, &success);
    if(success) {
        wp->e = malloc(strlen(e) + 1);
        strcpy(wp->e, e);
        wp->res = res;
    } else {
        Log_warn("Set wp expr fail: %s\n", e);
        return -1;
    }

    // find final head
    edb_wp_t* tmp = head;
    if (tmp == NULL) {
        head = wp;
    } else {
        while (tmp->next != NULL) {
        tmp = tmp->next;
        }
        tmp->next = wp;
    }
    wp->next = NULL;
    Log_info("Set wp (expr: %s) at NO.%d", wp->e, wp->NO);
    return wp->NO;
}

void edb_wp_free(int n) {
    if(n < 0 || n >= EDB_NWP) {
        Log_info("Invalid wp idx: %d", n);
        return;
    }
    edb_wp_t *wp = &wp_pool[n];

    // find in head
    edb_wp_t* tmp = head;
    if(tmp == NULL) {
        Log_info("No wp to free: %d", n);
        return;
    } else if (tmp == wp) {
        head = wp->next;
    } else {
        while (tmp->next != NULL) {
        if (tmp->next == wp) {
            tmp->next = wp->next;
            break;
        }
        tmp = tmp->next;
        }
    }
    // find final free_
    tmp = free_;
    if (tmp == NULL) {
        free_ = wp;
    } else {
        while (tmp->next != NULL) {
            tmp = tmp->next;
        }
        tmp->next = wp;
    }
    wp->next = NULL;
    Log_info("Free wp (expr: %s) at NO.%d", wp->e, wp->NO);
    free(wp->e);
    wp->e = NULL;
    wp->res = 0;
}

void edb_wp_info(int n) {
    if(n >= 0 && n < EDB_NWP) {
        edb_wp_t *wp = &wp_pool[n];
        EDB_INFO("- [%2d] expr: %s, res: %lu\n", wp->NO, wp->e, wp->res);
    } else {
        edb_wp_t* tmp = head;
        EDB_INFO("[Busy wp]: \n");
        while (tmp != NULL) {
            EDB_INFO("- [%2d] expr: %s, res: %lu\n", tmp->NO, tmp->e, tmp->res);
            tmp = tmp->next;
        }
    }
    return;
}

int edb_wp_scan(bool *change) {
    edb_wp_t* tmp = head;
    *change = false;
    int n = -1;
    while (tmp != NULL) {
        // deal with expr
        bool success;
        char* e = malloc(strlen(tmp->e) + 1);
        strcpy(e, tmp->e);
        uint64_t res = expr(e, &success);
        if(success) {
            if (tmp->res != res)  {
                Log_info("Scan wp (%s) change: %lu -> %lu", tmp->e, tmp->res, res);
                tmp->res = res;
                *change = true;
                n = tmp->NO;
                break;
            }
        } else {
            Log_warn("Scan wp expr fail: %s", e);
        }
        tmp = tmp->next;
    }
    return n;
}


// ==================================================================================== //
//                                    edb: FrameWork    
// ==================================================================================== //


static int edb_parse(int argc, char *argv[]) {
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
        case 'b': edb.is_batch_mode = true; break;
        case 'p': sscanf(optarg, "%d", &edb.diff_port); break;
        case 'l': edb.log_file = optarg; break;
        case 'd': edb.diff_file = optarg; break;
        case  1 : edb.model_file = optarg; return 0;
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

static void edb_welcome() {
    printf("Start time: %s %s\n", __TIME__, __DATE__);
    printf("Welcome to EDB %s!\n", _BLUE(STR(EDB_DEVICE)));
    printf("For help, type `help`\n");
}

void edb_init(int argc, char *argv[]) {
    edb_parse(argc, argv);
    srand(get_time_internal());
    edb_regex_init();
    edb_model_init();
    edb_wp_pool_init();
    edb_welcome();
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
    { "info", "Info of [g|w]"                                     , cmd_info },
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


static bool edb_trace(uint64_t* dnpc) {
    bool change;
    int n = edb_wp_scan(&change);
    if (change) {

    }
    return change;
}

static int cmd_c(UNUSED char *args) {
    graph_run(edb.ctx->graph);
    return 0;
}

static int cmd_si(char *args) {
    char *sub = strtok(args, " ");
    uint64_t n = 1;
    if (sub != NULL) {
        n = strtol(sub, NULL, 10);   
    }
    graph_step(edb.ctx->graph, n);
    return 0;
}

static int cmd_info(char *args) {
    char *sub = strtok(args, " ");
    if (sub == NULL) {
        EDB_INFO("Usage: info [g|w]\n");
    } else {
        if (strcmp(sub, "g") == 0) {
            graph_display();
        } else if (strcmp(sub, "w") == 0) {
            edb_wp_info(-1);
        } else {
            EDB_INFO("Usage: info [g|w]\n");
        }
    }
    return 0;
}

static int cmd_x(char *args) {
    char *sub = strtok(args, " ");
    if (sub == NULL) {
        EDB_INFO("Usage: x [N] [expr]\n");
    } else {
        /// TODO: implement x
    }
    return 0;
}

static int cmd_p(char *args) {
    char *sub = strtok(args, " ");
    if (sub == NULL) {
        EDB_INFO("Usage: p [expr]\n");
    } else {
        bool success;
        uint64_t res = expr(sub, &success);
        if (success) EDB_INFO("%s = %lu\n", sub, res);
    }
    return 0;
}
static int cmd_tsp(char *args) {
    char *sub = strtok(args, " ");
    if (sub == NULL) {
        EDB_INFO("Usage: tsp [file]\n");
    } else {
        // 1. 读入文件
        FILE *fp = fopen(sub, "r");
        if (fp == NULL) {
            EDB_INFO("Open file %s failed.\n", sub);
            return 0;
        }
        // 2. 读取每一行格式：`ref, exprisson` `int char*`
        char line[1024];
        while (fgets(line, 1024, fp) != NULL) {
            char *p = line;
            uint64_t ref = atoi(strtok(p, " "));
            char *exprisson = strtok(NULL, " ");
            // 删除换行符
            if (exprisson[strlen(exprisson) - 1] == '\n') {
                exprisson[strlen(exprisson) - 1] = '\0';
            }
            // 判断是否正确
            bool success;
            uint64_t res = expr(exprisson, &success);
            if (success && res == ref) {
                EDB_INFO(_GREEN("PASS")" %lu == %lu = %s\n",  res, ref, exprisson);
            } else if (success && res != ref) {
                EDB_INFO(_RED("FAIL")" %lu != %lu = %s\n",  res, ref, exprisson);
            }
        }
    }
    return 0;
}

static int cmd_w(char *args) {
    char *sub = strtok(args, " ");
    if (sub == NULL) {
        EDB_INFO("Usage: w [expr]\n");
    } else {
        edb_wp_new(sub);
    }
    return 0;
}

static int cmd_d(char *args) {
    char *sub = strtok(args, " ");
    if (sub != NULL) {
        UNUSED int n = atoi(sub);
        edb_wp_free(n);
    } else {
        EDB_INFO("Usage: d [n]\n");
    }
    return 0;
}

static int cmd_q(UNUSED char* args) {
    return -1;
}

static int cmd_help(UNUSED char *args){
    /* extract the first argument */
    char *arg = strtok(NULL, " ");
    size_t i;
    if (arg == NULL) {
        /* no argument given */
        for (i = 0; i < EDB_NCMD; i ++) {
            EDB_INFO("   %s - %s\n", cmd_table[i].name, cmd_table[i].description);
        }
    } else {
        for (i = 0; i < EDB_NCMD; i ++) {
            if (strcmp(arg, cmd_table[i].name) == 0) {
                EDB_INFO("   %s - %s\n", cmd_table[i].name, cmd_table[i].description);
                return 0;
            }
        }
        EDB_INFO("Unknown command `%s`", arg);
    }
    return 0;
}

void edb_loop() {
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
        for (i = 0; i < EDB_NCMD; i ++) {
            if (strcmp(cmd, cmd_table[i].name) == 0) {
                if (cmd_table[i].handler(args) < 0) {
                    linenoiseFree(line);
                    return;
                }
                break;
            }
        }
        if (i == EDB_NCMD) {
            EDB_INFO("Unknown command `%s`", cmd);
        }
        linenoiseFree(line);
    }
}

static void edb_diff_loop() {
    // 1. Creating socket file desc
    if((edb.listen = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        Log_err("socket failed!");
        exit(EXIT_FAILURE);
    }
    // 2. Forcefully attaching socket to the port EDB_DIFF_PORT
    int opt = 1;
    struct sockaddr_in address;
    socklen_t addrlen = sizeof(address);
    if(setsockopt(edb.listen, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        Log_err("setsockopt failed!");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(EDB_DIFF_POST);
    // 3. Forcefully binding socket to the port EDB_DIFF_PORT
    if(bind(edb.listen, (struct sockaddr*)&address, sizeof(address)) < 0) {
        Log_err("bind failed!");
        exit(EXIT_FAILURE);
    }
    Log_info("EDB Server listen port: %u", EDB_DIFF_POST);
    if(listen(edb.listen, 3) < 0) {
        Log_err("listen failed!");
        exit(EXIT_FAILURE);
    }
    while(1) { // Server Loop
        if((edb.connect = accept(edb.listen, (struct sockaddr*)&address, &addrlen)) < 0) {
            Log_err("accept failed!");
            exit(EXIT_FAILURE);
        }
        // subtract 1 for the null
        char buffer[EDB_BUFFER_SIZE] = { 0 };
        ssize_t valread = read(edb.connect, buffer, EDB_BUFFER_SIZE - 1);
        // recevice & send message
        if(buffer) {
            Log_info("Client cmd: %s", buffer);
            char *buffer_end = buffer + strlen(buffer);
            char *cmd = strtok(buffer, " ");
            if(cmd == NULL) {
                continue;
            }
            char *args = cmd + strlen(cmd) + 1;
            if(args >= buffer_end) {
                args = NULL;
            }
            size_t i;
            for(i = 0; i < EDB_NCMD; i++) {
                if(strcmp(buffer, cmd_table[i].name) == 0) {
                    if(cmd_table[i].handler(args) < 0) {
                        char* bye = "Bye from server";
                        send(edb.connect, bye, strlen(bye), 0);
                        Log_info("EDB Server quit");
                        close(edb.connect);
                        return;
                    }
                    break;
                }
            }
            if(i == EDB_NCMD) {
                EDB_INFO("Unknown command `%s`", cmd);
            }
        }
        // closing the connected socket
        close(edb.connect);
    }
    // closing the listening socket
    close(edb.listen);
    return;
}

void edb_exit() {
    if(edb.ctx) {
        printf("Unload model: %s(%u Byte) success!\n", edb.model_file, edb.ctx->model_size);
        edb.sez->unload(edb.ctx);
    }
    serializer_free(edb.sez);
    device_unreg(STR(EDB_DEVICE));
}

#undef EDB_NREGEX
#undef EDB_NCMD
#undef EDB_NWP

// ==================================================================================== //
//                                    edb: Prog Entry                                      
// ==================================================================================== //

int main(int argc, char *argv[]) {
    edb_init(argc, argv);
#ifdef EDB_DIFF_SERVER
    edb_diff_loop();
#else
    edb_loop();
#endif  // EDB_DIFF_SERVER
    edb_exit();
    return 0;
}