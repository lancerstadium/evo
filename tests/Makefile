
.PHONY: all run clean bare-rv32 emu-rv32

SRC_TEST 	:= $(wildcard *_tests.c)
TEST_TRG 	:= $(patsubst %.c,%,$(SRC_TEST))
INCDIRS		:= -I. -I../include
LIBDIRS		:= -L../build
LIBS 		:= -levo -lX11 -lm
LOG_TRG		:= tests.log
LOG_ENB     := 1
# PERF_TOOL	:= /usr/bin/time -v

all: $(TEST_TRG)

%: %.c
	@echo [CC] $<
ifeq ($(CROSS_COMPILE),riscv64-linux-gnu-)
	@echo [RV] $@
	@$(CC) $(INCDIRS) $(LIBDIRS) -DSOB_CLR_OFF -ffreestanding -nostartfiles -Wl,-Tbare.ld -O2 -o $@ $< $(LIBS) -static
else
	$(CC) $(INCDIRS) $< -o $@ $(LIBDIRS) $(LIBS)
endif


run: $(TEST_TRG)
	@echo;
	@for target in $(TEST_TRG); do \
		echo "[Test: $$target] >>>"; \
		$(PERF_TOOL) ./$$target 2> $(LOG_TRG) >&1; \
if [ $(LOG_ENB) -eq 1 ]; then \
		echo "[Msg: $$target] --------------------------------------"; \
		cat $(LOG_TRG); \
		echo "--------------------------------------------------------"; \
fi; \
	echo; \
	done

bare-rv32:
# riscv64-unknown-elf-objcopy -j .text -O binary --change-addresses=0x80000000 bare_tests bare_tests.bin
	riscv64-unknown-elf-objcopy -O binary -R .note.gnu.build-id -R .comment -R .riscv.attributes bare_tests bare_tests.bin
	riscv64-unknown-elf-objcopy -I binary -O ihex bare_tests.bin bare_tests.hex

check-rv32:
	riscv64-unknown-elf-objdump -h bare_tests

emu-rv32:
	qemu-system-riscv32 -nographic -machine virt -display none -bios none -serial mon:stdio -kernel bare_tests.bin -device loader,file=bare_tests.bin,addr=0x80000000


clean:
	@rm -f $(TEST_TRG)