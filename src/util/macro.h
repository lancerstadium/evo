

// ==================================================================================== //
//                                    util: macro
// ==================================================================================== //

#ifndef UTIL_MACRO_H
#define UTIL_MACRO_H

// ==================================================================================== //
//                                    macro: Size
// ==================================================================================== //

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define GET_ARR_LEN(arr) (sizeof(arr) / sizeof(arr[0]))

#define ARG_FIRST(first, ...) first
#define ARG_OTHER(first, ...) , ## __VA_ARGS__

#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

#define REP1(M, a1) M (a1)
#define REP2(M, a1, a2) M (a1) REP_SEP M (a2)
#define REP3(M, a1, a2, a3) REP2 (M, a1, a2) REP_SEP M (a3)
#define REP4(M, a1, a2, a3, a4) REP3 (M, a1, a2, a3) REP_SEP M (a4)
#define REP5(M, a1, a2, a3, a4, a5) REP4 (M, a1, a2, a3, a4) REP_SEP M (a5)
#define REP6(M, a1, a2, a3, a4, a5, a6) REP5 (M, a1, a2, a3, a4, a5) REP_SEP M (a6)
#define REP7(M, a1, a2, a3, a4, a5, a6, a7) REP6 (M, a1, a2, a3, a4, a5, a6) REP_SEP M (a7)
#define REP8(M, a1, a2, a3, a4, a5, a6, a7, a8) REP7 (M, a1, a2, a3, a4, a5, a6, a7) REP_SEP M (a8)
#define REP_SEP ,






#endif // UTIL_MACRO_H