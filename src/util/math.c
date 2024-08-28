#include <evo/util/math.h>
#include <stdlib.h>

int imin(int a, int b) {
    return a <= b ? a : b;
}

int imax(int a, int b) {
    return a >= b ? a : b;
}

int min_abs(int a, int b) {
    return imin(abs(a), abs(b));
}

int max_abs(int a, int b) {
    return imax(abs(a), abs(b));
}

static int solve_gcd(int large, int small) {
    int val = large % small;
    return 0 == val ? small : gcd(small, val);
}

int gcd(int a, int b) {
    if (0 == a || 0 == b)
        return 0;

    return solve_gcd(max_abs(a, b), min_abs(a, b));
}

int lcm(int a, int b) {
    if (0 == a || 0 == b)
        return 0;

    return abs(a * b) / solve_gcd(max_abs(a, b), min_abs(a, b));
}

int align(int value, int step) {
    const int mask = ~(abs(step) - 1);
    return (value + step) & mask;
}

void* align_address(void* address, int step) {
    const size_t mask = ~(abs(step) - 1);
    return (void*)((size_t)address & mask);
}