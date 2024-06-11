#undef CONFIG_NO_DEBUG
#ifndef _Minunit_h_
#define _Minunit_h_

#include <stdio.h>
#include <stdlib.h>
#include <util/macro.h>
#include <util/log.h>
#include <time.h>

// 1e3: us, 1e6: ms, 1e9: s
#define MU_TIMESCALE 1e6
// 1: Open assert message, 0: No assert message
#define MU_ASTMSG 0

#define mu_suite_start() char *message = NULL

#define mu_time_start() \
    clock_gettime(CLOCK_MONOTONIC, &start_time); \

#define mu_time_end(scale) \
    clock_gettime(CLOCK_MONOTONIC, &end_time); \
    time_scale = (scale == 1e3) ? "us" : (scale == 1e6) ? "ms" : "s"; \
    time_taken = ((end_time.tv_sec - start_time.tv_sec) * 1e9 + end_time.tv_nsec - start_time.tv_nsec) / scale; \
    if (end_time.tv_nsec < start_time.tv_nsec) { \
        time_taken += 1; \
    } \
    time_total_taken += time_taken;


#define mu_cnt_res(res) ((res == NULL) ? (tests_pass++) : (tests_fail++))
#define mu_run_res(res) ((res == NULL) ? _green("PASS") : _red("FAIL"))
#define mu_msg(...) \
    do {\
        char message[64]; \
        snprintf(message, 64, ARG_FIRST(__VA_ARGS__) ARG_OTHER(__VA_ARGS__)); \
        printf("│  │ " _yellow("msg: ") _grey("%-38s") " │\n", message); \
    } while(0)


#if MU_ASTMSG == 0

#define mu_ast(test, message) \
    do {\
        if (!(test)) {\
            printf("│  │ " _yellow("ast: ") _red("%-38s") " │\n", #test); \
            mu_msg(message); \
            log_error(message); \
            return message; \
        } \
    } while(0)
#else

#define mu_ast(test, message) \
    do {\
        if (!(test)) {\
            printf("│  │ " _yellow("ast: ") _grey("%-25s ") _red("%-12s") " │\n", message, #test); \
            log_error(message); \
            return message; \
        } \
        else {\
            printf("│  │ " _yellow("ast: ") _green("%-38s") " │\n", #test); \
        } \
    } while(0)

#endif


#define mu_run_test(test) \
    log_debug("\n──────%s", " Sub: " _blue(#test)); \
    mu_time_start(); \
    message = test(); \
    tests_run++; \
    mu_cnt_res(message); \
    mu_time_end(MU_TIMESCALE); \
    printf("│  ├── " _mag("%-2d ") _blue("%-18s") _cyan("%12.4f %2s") " %s │\n", tests_run, #test, time_taken, time_scale, mu_run_res(message));\
    log_debug("total exec %.3f %2s", time_taken, time_scale); \
    if (message) return message;

#define RUN_TESTS(name) \
int main(int, char *argv[]) {\
    log_debug("\n\n────── Run: " _blue("%s"), argv[0]);\
    printf("┌────────────────────────────────────────────────┐\n");\
    printf("│ Test: " _blue("%-40s") " │\n", argv[0]);\
    char *result = name();\
    printf("│ Sum: " _mag("%-2d ") "[%2d " _green("PASS") " %2d " _red("FAIL") "] " _cyan("%12.4f %2s") " %s │\n", tests_run, tests_pass, tests_fail, time_total_taken, time_scale, mu_run_res(result));\
    printf("├────────────────────────────────────────────────┤\n");\
    if (result == NULL) { \
        printf("│ " _cyan("%-3s ") _blue("%-37s ") "%s │\n", "Res" , argv[0], _green("PASS")); \
    } else { \
        printf("│ " _cyan("%-3s ") _blue("%-37s ") "%s │\n", "Res" , argv[0], _red("FAIL")); \
        printf("│ " _cyan("%-3s ") _red("%-42s") " │\n", "Msg" , result); \
        printf("│ %-3s %-51s │\n", _cyan("Log"), _yellow("tests/tests.log")); \
    } \
    printf("└────────────────────────────────────────────────┘\n"); \
    exit(result != 0);\
}


static int tests_run = 0;
static int tests_fail = 0;
static int tests_pass = 0;
static struct timespec start_time;
static struct timespec end_time;
static double time_taken;
static double time_total_taken = 0;
static char* time_scale = NULL;

#endif