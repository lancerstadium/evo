

// ==================================================================================== //
//                                       Include
// ==================================================================================== //

#include "core/core.h"
#include <time.h>
#include <unistd.h>

// ==================================================================================== //
//                                   Apps Define
// ==================================================================================== //

ap_def_callback(evoc);
ap_def_callback(evo_hello);
ap_def_callback(evo_test);


void arg_parser(int argc, char *argv[], char *envp[]);

// ==================================================================================== //
//                                Commands & Args Define
// ==================================================================================== //

// 定义参数
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
    ap_init_parser("evo - Evolvable Programming Language    \n"
                   "                                        \n"
                   "   $$$$$$\\  $$\\    $$\\  $$$$$$\\     \n"
                   "  $$  __$$\\ \\$$\\  $$  |$$  __$$\\    \n"
                   "  $$$$$$$$ | \\$$\\$$  / $$ /  $$ |     \n"
                   "  $$   ____|  \\$$$  /  $$ |  $$ |      \n"
                   "  \\$$$$$$$\\    \\$  /   \\$$$$$$  |   \n"
                   "   \\_______|    \\_/     \\______/     \n"
                   "                                        \n", NULL);
    // 添加命令
    ap_add_command("default"    , "evo compiler"            , "This is usage."  , evoc              , default_args);
    ap_add_command("hello"      , "Print `Hello, World!`."  , "This is usage."  , evo_hello         , default_args);
    ap_add_command("test"       , "Unit test"               , "This is usage."  , evo_test          , test_args);
    // 开始解析
    ap_do_parser(argc, argv, envp);
}



ap_def_callback(evoc){

    if (argc == 1) {
        log_warn("usage: %s <file>", argv[0]);
        return;
    }

    LOG_TAG
    char* arg = argv[1];
    int res = compile_file(arg, NULL, 0);

    if(res == COMPILER_FILE_OK) {
        log_info("Compile success!");
    }else if(res == COMPILER_FILE_ERROR) {
        log_error("Compile failed!");
    }else {
        log_error("Unknown res code %d", res);
    }

}


ap_def_callback(evo_hello) {
    printf("[EVO]: hello, world!\n");
}

void toml_test(const char* filename) {

    // 1. Read and parse toml file
    FILE* fp;
    char errbuf[200];
    fp = fopen(filename, "r");
    toml_table_t* conf = toml_parse_file(fp, errbuf, sizeof(errbuf));
    fclose(fp);
    // 2. Traverse to a table.
    toml_table_t* server = toml_table_in(conf, "server");
    log_error_if(!fp, "cannot open %s", filename);
    // 3. Extract values
    toml_datum_t host = toml_string_in(server, "host");
    log_error_if(!host.ok, "cannot read server.host");
    toml_array_t* portarray = toml_array_in(server, "port");
    log_error_if(!portarray, "cannot read server.port");
    toml_array_t* srcarray = toml_array_in(server, "src");
    printf("host: %s\n", host.u.s);
    printf("port: ");
    for (int i = 0; ; i++) {
        toml_datum_t port = toml_int_at(portarray, i);
        if (!port.ok) break;
        printf("%d ", (int)port.u.i);
    }
    printf("\n");
    printf("src:  ");
    for (int i = 0; ; i++) {
        toml_datum_t src = toml_string_at(srcarray, i);
        if(!src.ok) break;
        printf("%s ", src.u.s);
    }
    printf("\n");
    // 4. Free memory
    free(host.u.s);
    toml_free(conf);

}

#define usec_to_msec(n) (1000*(n))
const char *rand_bit()
{
	return rand() % 2 ? ANSI_FMT("1" ANSI_WHITE) : ANSI_FMT("0" ANSI_RED);
}

const char *eq()
{
	return "█";
}

const char *eq2() {
    return "─"; // ━
}

const char *eq3() {
    return "═";
}

void widget_test() {
    struct progressbar *bar;
	int i;

	srand(time(NULL));

	bar = progressbar_new("READ ", 1000, rand_bit);
	for (i = 0; i < 1000; ++i) {
		usleep(usec_to_msec(1));
		progressbar_inc(bar);
	}
	progressbar_finish(bar);
	
	bar = progressbar_new("ERASE", 200, eq3);
	for (i = 0; i < 200; ++i) {
		usleep(usec_to_msec(12));
		progressbar_inc(bar);
	}
	progressbar_finish(bar);
	
	bar = progressbar_new("WRITE", 200, eq2);
	for (i = 0; i < 200; ++i) {
		usleep(usec_to_msec(10));
		progressbar_inc(bar);
	}
	progressbar_finish(bar);
}


void log_test() {
    log_trace("nihao");
    log_info("nihao");
    log_debug("nihao");
    log_warn("nihao");
    log_error("nihao");
    // log_assert(1, "nihao");
    // log_fatal("nihao");
}


void suit_test_01() {
    ut_print_test();
}

ap_def_callback(evo_test) {
    

    ap_arg_t* arg_o = ap_get("output");
    log_assert(arg_o != NULL, "arg_o should not NULL");
    log_assert(arg_o->value == NULL, "");

    if (!arg_o->value) {
        printf("Output Init: %s\n", (!arg_o->value) ? arg_o->init.s : arg_o->value);
    }
    
    int q_l;
    if(!ap_get("quiet")->value) {
        q_l = ap_get("quiet")->init.i;
    } else {
        q_l = atoi(ap_get("quiet")->value);
    }
    printf("set quiet: %d\n", q_l);
    ut_set_quiet(q_l);

    // toml_test("demo.toml");
    widget_test();
    log_test();

    
    suit_test_01();


}

// ==================================================================================== //
//                                  Proc Entry: evo
// ==================================================================================== //

int main(int argc, char *argv[], char *envp[]) {
    arg_parser(argc, argv, envp);
    return 0;
}