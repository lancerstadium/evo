/**
 * @file unittest.h
 * @author lancer (lancerstadium@163.com)
 * @brief 单元测试头文件
 * @version 0.1
 * @date 2023-12-28
 *
 * @copyright Copyright (c) 2023
 * @note 参考项目：[Gitee | MiniUnit](https://gitee.com/zhuangbo/MiniUnit)
 *
 * A small unit test framework for C/C++.
 *
 * Origin author: Bo Zhuang <sdzhuangbo@hotmail.com>
 * Features:
 *   - `ut_assert(expr)` assertion fail if `expr` false
 *   - `ut_assert(expr, message)` assertion with message
 *   - `ut_assert(expr, message, args...)` assertion with message and args
 *   - `ut_run_test(test)` to run a test function in form `int f()`
 *                         return 0 if passed
 *   - `ut_test_results()` to display the test results
 *   - `#define MU_NOCOLOR` if ANSI escape code not supported
 *
 * For example,
 * ```c
 * #include "unittest.h"
 *
 * int test_one() {
 *   ut_assert(2 + 2 == 4);
 *   return 0; // 0 means test passed
 * }
 *
 * int test_two() {
 *   int a = 3, b = 5;
 *   ut_assert(a == 3);
 *   ut_assert(b == 5, "b is 5");
 *   ut_assert(a + b == 7, "should be %d", a + b); // fail
 *   return 0;
 * }
 *
 * int main()
 * {
 *   ut_run_test(test_one);
 *   ut_run_test(test_two);
 *
 *   ut_test_results();
 *
 *   return 0;
 * }
 * ```
 */

#ifndef UNIT_TEST_H
#define UNIT_TEST_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include "color.h"

/* Count of args */
#define VA_NARG(args...) VA_NARG_(0, ##args, VA_RSEQ_N())
#define VA_NARG_(args...) VA_ARG_N(args)
#define VA_ARG_N(_0, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, N, ...) N
#define VA_RSEQ_N() 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

/* The first arg */
#define VA_FIRST(args...) VA_FIRST_(args)
#define VA_FIRST_(F, ...) F

/* The first arg in string */
#define VA_FIRST_STR(args...) VA_FIRST_STR_(args)
#define VA_FIRST_STR_(F, ...) #F

/* The rest args */
#define VA_REST_(F, args...) args
#define VA_REST(args...) VA_REST_(args)

#ifdef __linux__
#define UT_FAIL "✘"
#define UT_PASS "✔"
#else /* for others */
#define UT_FAIL "FAIL"
#define UT_PASS "PASS"
#endif /* __linux__ */

struct unittest_t
{
    int n_test; /* number of tests */
    int n_pass; /* number of tests passed */
    int n_fail; /* number of tests failed */
    int u_t;
    int u_p;
    int u_f;
    int flag;
    int quiet;
};

static struct unittest_t ut = {
    .n_fail = 0,
    .n_pass = 0,
    .n_test = 0,
    .u_f = 0,
    .u_p = 0,
    .u_t = 0,
    .flag = 0,
    .quiet = 0
};

#define ut_set_quiet(num) \
    ut.quiet = num;

#define ut_assert(test...)                                                                      \
    do                                                                                          \
    {                                                                                           \
        ut.u_t++;                                                                               \
        if (!(VA_FIRST(test)))                                                                  \
        {                                                                                       \
            ut.u_f++;                                                                           \
            ut.flag = 1;                                                                        \
            if (ut.quiet <= 3)                                                                  \
            {                                                                                   \
                printf("\n"                                                                     \
                       "|  - " _bred("FAIL %d") " " _black("%s:%d") ": '" _yellow("%s") "' ",   \
                       ut.u_t,                                                                  \
                       __FILE__, __LINE__, VA_FIRST_STR(test));                                 \
                if (VA_NARG(test) == 1 || ut.quiet > 1)                                         \
                    printf(_bred(UT_FAIL));                                                     \
                else                                                                            \
                    printf(_bred(UT_FAIL) "\n"                                                  \
                                          "|     : " VA_REST(test));                            \
            }                                                                                   \
            break;                                                                              \
        }                                                                                       \
        else                                                                                    \
        {                                                                                       \
            ut.u_p++;                                                                           \
            if (ut.quiet <= 0)                                                                  \
            {                                                                                   \
                printf("\n"                                                                     \
                       "|  - " _bgreen("PASS %d") " " _black("%s:%d") ": '" _yellow("%s") "' ", \
                       ut.u_t,                                                                  \
                       __FILE__, __LINE__, VA_FIRST_STR(test));                                 \
                if (VA_NARG(test) == 1 || ut.quiet > 1)                                         \
                    printf(_bgreen(UT_PASS));                                                   \
                else                                                                            \
                    printf(_bgreen(UT_PASS) "\n"                                                \
                                            "|     : " VA_REST(test));                          \
            }                                                                                   \
            break;                                                                              \
        }                                                                                       \
    } while (0)

#define ut_dec_test(testname) \
    int testname##_test()

#define ut_def_test(testname, ...)                        \
    int testname##_test()                                 \
    {                                                     \
        ut.flag = 0;                                      \
        ut.u_t = 0, ut.u_p = 0, ut.u_f = 0;               \
        __VA_ARGS__                                       \
        if (ut.quiet < 3)                                 \
        {                                                 \
            printf("\n"                                   \
                   "|-" _bgreen(" %d " UT_PASS) " - ",    \
                   ut.u_p);                               \
            printf(_bred("%d " UT_FAIL), ut.u_f);         \
            printf(" - " _bold("%d all") " \n|", ut.u_t); \
        }                                                 \
        return ut.flag;                                   \
    }

#define ut_run_test(testname)                                     \
    do                                                            \
    {                                                             \
        if (ut.n_test == 0 && ut.quiet < 5)                       \
        {                                                         \
            printf("\n"                                           \
                   "|============== Unit tests ==============="); \
        }                                                         \
        ++ut.n_test;                                              \
        if (ut.quiet < 5)                                         \
        {                                                         \
            printf("\n"                                           \
                   "|> " _bold("%s") _black(" %s:%d "),           \
                   #testname, __FILE__, __LINE__);                \
        }                                                         \
        if (!testname##_test())                                   \
        {                                                         \
            ++ut.n_pass;                                          \
            if (ut.quiet > 2 && ut.quiet < 5)                     \
            {                                                     \
                printf(_bgreen(UT_PASS));                         \
            }                                                     \
        }                                                         \
        else                                                      \
        {                                                         \
            ++ut.n_fail;                                          \
            if (ut.quiet > 3 && ut.quiet < 5)                     \
            {                                                     \
                printf(_bred(UT_FAIL));                           \
            }                                                     \
        }                                                         \
    } while (0)

#define ut_print_test()                                                             \
    do                                                                              \
    {                                                                               \
        if (ut.quiet < 6)                                                           \
        {                                                                           \
            printf("\n"                                                             \
                   "|============== Test Result ==============");                   \
            printf("\n"                                                             \
                   "| " _bgreen("%d " UT_PASS) " and ",                             \
                   ut.n_pass);                                                      \
            printf(_bred("%d " UT_FAIL), ut.n_fail);                                \
            printf(" in " _bold("%d TEST(S)"), ut.n_test);                          \
            if (ut.n_pass == ut.n_test)                                             \
                printf("\n"                                                         \
                       "|============" _green(" ALL TESTS PASSED ") "==========="); \
            else                                                                    \
                printf("\n"                                                         \
                       "|===========" _red(" %d TEST(S) FAILED ") "============"    \
                                                                  "\n",             \
                       ut.n_fail);                                                  \
        }                                                                           \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif /* UNIT_TEST_H */
