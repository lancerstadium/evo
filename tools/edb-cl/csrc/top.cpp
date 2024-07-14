// verilator
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vtop.h"

// glibc
#include <stdlib.h>
#include <assert.h>

// sys
#include <sys/time.h>

// edb

// =================================================== //
//                    Environment
// =================================================== //

VerilatedContext* contextp;
Vtop* top;
VerilatedVcdC* tfp;

void single_cycle_vcd_record() {
    tfp->dump(contextp->time());
    contextp->timeInc(1);
}

void single_cycle() {
    top->clk = 1;
    top->eval();
    single_cycle_vcd_record();
    top->clk = 0;
    top->eval();
    single_cycle_vcd_record();
}

void reset(int n) {
    top->rst = 1;
    while (n-- > 0) single_cycle();
    top->rst = 0;
}


void elem_init() {
    top->A = rand() % 2001 - 1000;
    top->B = rand() % 2001 - 1000;
}

void elem_display() {
    printf("A = %5d, B = %5d, C = %5d, C_hw = %5d\n", 
        top->A, 
        top->B, 
        top->A + top->B, 
        top->C);
}

// =================================================== //
//                       Main
// =================================================== //

int main(int argc, char **argv, char **env) {

    srand((unsigned int)time(NULL));
    contextp = new VerilatedContext;
    contextp->commandArgs(argc, argv);
    top = new Vtop{contextp};

    // VCD wave initialization
    tfp = new VerilatedVcdC;            // Initialize VCD pointer
    contextp->traceEverOn(true);        // Enable tracing
    top->trace(tfp, 0);

    // VCD wave setting
    tfp->open("top.vcd");              // VCD file path
    tfp->set_time_unit("ns");           // Set time unit to nanoseconds
    reset(10);

    // char *cmd = NULL;
    // int ret = 0;
    // char buffer[1024] = {0};
    // do {
    //     if(cmd) {
    //         ret = edb_client_send(cmd, buffer);
    //         linenoiseFree(cmd);
    //         if(strcmp(cmd, "si")) {
    //             printf("-------- hw --------\n");
    //             elem_init();
    //             single_cycle();
    //             elem_display();
    //             printf("-------- hw --------\n");
    //         }
    //     }
    // } while(((cmd = linenoise("(EDB) ")) != NULL) && (ret == 0));

    // VCD wave dump
    tfp->close();
    // End of simulation
    top->final();

    delete tfp;
    delete top; 
    delete contextp;
    return 0;
}