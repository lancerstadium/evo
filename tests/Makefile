
.PHONY: all run clean

SRC_TEST 	:= $(wildcard *_tests.c)
TEST_TRG 	:= $(patsubst %.c,%,$(SRC_TEST))
INCDIRS		:= -I . -I ../evo
LIBDIRS		:= -L ../build
LIBS 		:= -levo -lm
LOG_TRG		:= tests.log
LOG_ENB     := 1

all: $(TEST_TRG)

%: %.c
	@echo [CC] $<
	@$(CC) $(INCDIRS) $(LIBDIRS) -o $@ $< $(LIBS) -static

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

clean:
	@rm -f $(TEST_TRG)