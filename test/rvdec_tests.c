#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "minunit.h"
#include <dec/rvdec.h>


static int test_impl(RvOptions opt, unsigned len, uint32_t inst_raw,
                     const char* exp_fmt) {
    RvInst inst;
    char fmt[128];
    int retval = rv_decode(len, (unsigned char*) &inst_raw, opt, &inst);
    if (retval == RV_PARTIAL)
        strcpy(fmt, "PARTIAL");
    else if (retval == RV_UNDEF)
        strcpy(fmt, "UNDEF");
    else
        rv_format(&inst, sizeof fmt, fmt);
    if ((retval < 0 || (unsigned) retval == len) && !strcmp(fmt, exp_fmt)) {
        mu_msg("OK: %s", fmt);
        return 0;
    }
    mu_msg("Failed case: %08" PRIx32, inst_raw);
    mu_msg("- Exp (%2zu): %s", sizeof inst_raw, exp_fmt);
    mu_msg("- Got (%2d): %s", retval, fmt);
    return -1;
}

#define test32(...) test_impl(RV_RV32, __VA_ARGS__)
#define test64(...) test_impl(RV_RV64, __VA_ARGS__)
#define test(...) test32(__VA_ARGS__) | test64(__VA_ARGS__)


char *test_decode(){
    unsigned failed = 0;
    failed |= test(4, 0x00000000, "UNDEF");
    failed |= test(4, 0x00054703, "lbu r14 r10");
    failed |= test(4, 0xfe043783, "ld r15 r8 -32");
    failed |= test(4, 0xfe043023, "sd r8 r0 -32");
    failed |= test(4, 0x00d71463, "bne r14 r13 8");
    failed |= test(4, 0xfe0718e3, "bne r14 r0 -16");
    failed |= test(4, 0x0ff67613, "andi r12 r12 255");
    failed |= test64(4, 0x0007879b, "addiw r15 r15");
    failed |= test(4, 0x00008067, "jalr r0 r1");
    failed |= test(4, 0x0700006f, "jal r0 112");
    failed |= test(4, 0x20a93c27, "fsd r18 r10 536");
    failed |= test64(4, 0xe20505d3, "fmv.x.d r11 r10");
    failed |= test64(4, 0xd2287553, "fcvt.d.l r10 r16");
    failed |= test(4, 0x02957553, "fadd.d r10 r10 r9");
    failed |= test(4, 0x420686d3, "fcvt.d.s r13 r13");
    failed |= test(4, 0x00100013, "addi r0 r0 1");

    failed |= test(2, 0x4601, "addi r12 r0"); // implicit 0 in printed output
    failed |= test(2, 0x002c, "addi r11 r2 8");
    failed |= test(2, 0x714d, "addi r2 r2 -336");
    failed |= test(2, 0x0521, "addi r10 r10 8");
    failed |= test(2, 0x1571, "addi r10 r10 -4");
    failed |= test(2, 0x00a8, "addi r10 r2 72");
    failed |= test32(2, 0x641c, "flw r15 r8 8");
    failed |= test(2, 0x87b6, "add r15 r0 r13");
    failed |= test(2, 0xc05c, "sw r8 r15 4");
    failed |= test64(2, 0x6582, "ld r11 r2");
    failed |= test64(2, 0xfa22, "sd r2 r8 304");
    failed |= test64(2, 0xc93e, "sw r2 r15 144");
    failed |= test64(2, 0x47c2, "lw r15 r2 16");
    failed |= test32(2, 0xe09c, "fsw r9 r15");
    failed |= test64(2, 0xe09c, "sd r9 r15");
    failed |= test(2, 0x050e, "slli r10 r10 3");
    failed |= test(2, 0xfe75, "bne r12 r0 -4");
    failed |= test(2, 0xa029, "jal r0 10");
    failed |= test(2, 0x78fd, "lui r17 -4096");
    failed |= test(2, 0x0001, "addi r0 r0"); /* C.ADDI is normally not allowed an imm=0, except with rd=0 encoding a NOP */
    return (failed ? "Some tests FAILED" : NULL);
}


char *all_tests() {
    mu_suite_start();
    mu_run_test(test_decode);
    return NULL;
}

RUN_TESTS(all_tests);