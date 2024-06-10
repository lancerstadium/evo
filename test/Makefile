
FIRST_GOAL := $(firstword $(MAKECMDGOALS))
SECOND_GOAL := $(word 2, $(MAKECMDGOALS))
THIRD_GOAL := $(word 3, $(MAKECMDGOALS))

PROG := $(if $(SECOND_GOAL),$(SECOND_GOAL),hello)

SRC_PROC := $(PROG).c
ELF_PROC := $(PROG).elf
BIN_PROC := $(PROG).bin


all:
	@echo "Gen RISC-V Insn."

rv32:
	riscv64-linux-gnu-gcc-12 -march=rv32i -mabi=ilp32 --static -nostdlib -ffreestanding -Wl,-Ttext=0 -O2 -o $(ELF_PROC) $(SRC_PROC)

dp32:
	riscv64-linux-gnu-objdump -D $(ELF_PROC)

bin:
	riscv64-linux-gnu-objcopy -j .text -O binary $(ELF_PROC) $(BIN_PROC)

clean:
	rm -f $(ELF_PROC) $(BIN_PROC)

.PHONY: all rv32 bin clean