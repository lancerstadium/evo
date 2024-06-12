
FIRST_GOAL := $(firstword $(MAKECMDGOALS))
SECOND_GOAL := $(word 2, $(MAKECMDGOALS))
THIRD_GOAL := $(word 3, $(MAKECMDGOALS))

PROG := $(if $(SECOND_GOAL),$(SECOND_GOAL),hello)

SRC_PROC := $(PROG).c

RV32_ELF := $(PROG)_rv32.elf
RV32_BIN := $(PROG)_rv32.bin

I386_ELF := $(PROG)_i386.elf
I386_BIN := $(PROG)_i386.bin

ARM_ELF := $(PROG)_arm.elf
ARM_BIN := $(PROG)_arm.bin


all:
	@echo "Gen RISC-V Insn."

rv32:
	riscv64-linux-gnu-gcc-12 -march=rv32i -mabi=ilp32 --static -nostdlib -ffreestanding -Wl,-Ttext=0 -O2 -o $(RV32_ELF) $(SRC_PROC)

i386:
# x86_64-linux-gnu-gcc-12 -march=i386 --static -nostdlib -ffreestanding -Wl,-Ttext=0 -O2 -o $(I386_ELF) $(SRC_PROC)
	gcc -m32 -ffreestanding -W -o $(I386_ELF) $(SRC_PROC)


rv32-dump:
	riscv64-linux-gnu-objdump -M no-aliases,numeric -D $(RV32_ELF)

rv32-bin:
	riscv64-linux-gnu-objcopy -j .text -O binary $(RV32_ELF) $(RV32_BIN)

i386-dump:
	x86_64-linux-gnu-objdump -M no-aliases,numeric -D $(I386_ELF)

i386-bin:
	x86_64-linux-gnu-objcopy -j .text -O binary $(I386_ELF) $(I386_BIN)



clean:
	rm -f *.elf *.bin

.PHONY: all rv32 bin clean