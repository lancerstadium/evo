#include <evo.h>


int main() {
    device_t *cpu = device_reg("cpu");
    device_unreg_dev(cpu);
    return 0;
}