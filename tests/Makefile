
.PHONY: all run clean bare-rv32 emu-rv32

SRC_TEST 	:= $(wildcard *_tests.c)
TEST_TRG 	:= $(patsubst %.c,%,$(SRC_TEST))
INCDIRS		:= -I. -I../include 
LIBDIRS		:= -L../build
LIBS		:= -levo -lm -lX11
LOG_TRG		:= tests.log
LOG_ENB     := 1
MEM_ENB		:= 0
# PERF_TOOL	:= /usr/bin/time -v
VALGRIND_TOOL := valgrind --tool=massif --time-unit=B --detailed-freq=10 --massif-out-file=massif.out

all: $(TEST_TRG)

%: %.c
	@echo [CC] $<
	@chmod 777 ../build/libevo.a
ifeq ($(CROSS_COMPILE),riscv32-unknown-elf-)
	@echo [RV] $@
	@$(CC) $(INCDIRS) -I/opt/riscv/riscv32-unknown-elf/include  $(LIBDIRS) -DSOB_CLR_OFF -ffreestanding -nostartfiles -Wl,-Tbare.ld -O2 -o $@ $< -levo -lm -static
else ifeq ($(CROSS_COMPILE),aarch64-linux-gnu-)
	$(INCDIRS) +=  -I/usr/aarch64-linux-gnu/include
	$(LIBDIRS) +=  -L/usr/aarch64-linux-gnu/lib/aarch64-linux-gnu/
	@echo [ARM] $@
	@$(CC) $(INCDIRS) --static $< -o $@ $(LIBDIRS) $(LIBS) -lX11
else
	$(CC) $(INCDIRS) $< -o $@ $(LIBDIRS) $(LIBS) -lcudart
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
if [ $(MEM_ENB) -eq 1 ]; then \
	echo "[Memory Analysis: $$target]"; \
	$(VALGRIND_TOOL) ./$$target > /dev/null 2>&1; \
	echo "--------------------------------------------------------------------------------------------"; \
	echo "Snapshot   Time (ms)   Heap Memory (KB)    Extra Heap Memory (KB)    Stack Memory (KB)"; \
	echo "--------------------------------------------------------------------------------------------"; \
	awk ' \
		BEGIN {freq_ghz = 1.0; factor = 1000 / (freq_ghz * 1e9); max_mem = 0; min_mem = 1e9; total_mem = 0; count = 0} \
		/^snapshot=/ {snapshot = substr($$0, index($$0, "=") + 1)} \
		/^time=/ {time = substr($$0, index($$0, "=") + 1) * factor} \
		/^mem_heap_B=/ {mem_heap = substr($$0, index($$0, "=") + 1) / 1024} \
		/^mem_heap_extra_B=/ {mem_heap_extra = substr($$0, index($$0, "=") + 1) / 1024} \
		/^mem_stacks_B=/ {mem_stacks = substr($$0, index($$0, "=") + 1) / 1024} \
		/^heap_tree=/ { \
			# 计算最大值、最小值和总和 \
			if (mem_heap > max_mem) max_mem = mem_heap; \
			if (mem_heap < min_mem) min_mem = mem_heap; \
			total_mem += mem_heap; \
			count++; \
			# 输出表格行 \
			printf "%-10s %-10.2f %-20.2f %-20.2f %-20.2f\n", snapshot, time, mem_heap, mem_heap_extra, mem_stacks; \
			snapshot = time = mem_heap = mem_heap_extra = mem_stacks = "" \
		} \
		END { \
			# 计算平均值并输出统计结果 \
			avg_mem = (count > 0) ? total_mem / count : 0; \
			printf "--------------------------------------------------------------------------------------------\n"; \
			printf "%-10s %-10s %-20.2f %-20s %-20s\n", "Max", "", max_mem, "", ""; \
			printf "%-10s %-10s %-20.2f %-20s %-20s\n", "Min", "", min_mem, "", ""; \
			printf "%-10s %-10s %-20.2f %-20s %-20s\n", "Avg", "", avg_mem, "", ""; \
		}' massif.out; \
	echo "--------------------------------------------------------------------------------------------"; \
	echo; \
fi; \
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