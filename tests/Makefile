
.PHONY: all run clean bare-rv32 emu-rv32

SRC_TEST 	:= $(wildcard *_tests.c)
TEST_TRG 	:= $(patsubst %.c,%,$(SRC_TEST))
INCDIRS		:= -I. -I../evo
LIBDIRS		:= -L../build
LIBS 		:= -levo -lm
LOG_TRG		:= tests.log
LOG_ENB     := 1

all: $(TEST_TRG)

%: %.c
	@echo [CC] $<
ifeq ($(CROSS_COMPILE),riscv64-linux-gnu-)
	@echo [RV] $@
	@$(CC) $(INCDIRS) $(LIBDIRS) -DSOB_CLR_OFF -DBARE_VIRT -ffreestanding -nostartfiles -Wl,-Tbare.ld -O2 -o $@ $< $(LIBS) -static
else
	@$(CC) $(INCDIRS) $(LIBDIRS) -o $@ $< $(LIBS) -static
endif


run: $(TEST_TRG)
	@echo;
	@for target in $(TEST_TRG); do \
		echo "[Test: $$target] >>>"; \
		./$$target 2> $(LOG_TRG) >&1; \
if [ $(LOG_ENB) -eq 1 ]; then \
		echo "[Msg: $$target] --------------------------------------"; \
		cat $(LOG_TRG); \
		echo "--------------------------------------------------------"; \
fi; \
	echo; \
	done

bare-rv32:
# riscv64-linux-gnu-gcc $(INCDIRS) $(LIBDIRS) $(LIBS) -DSOB_CLR_OFF --static -ffreestanding -nostdlib -Wl,-Tbare.ld -O2 bare_tests.c -o bare_tests
	riscv64-unknown-elf-objcopy -j .text -O binary --change-addresses=0x80000000 bare_tests bare_tests.bin

emu-rv32:
	qemu-system-riscv32 -nographic -machine virt -display none -bios none -serial mon:stdio -kernel bare_tests.bin -device loader,file=bare_tests.bin,addr=0x80000000


clean:
	@rm -f $(TEST_TRG)