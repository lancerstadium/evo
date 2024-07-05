
.PHONY: all clean

SRC_TEST := $(wildcard *_tests.c)
TEST_TRG := $(patsubst %.c,%,$(SRC_TEST))
INCDIRS		:= -I . -I ../src
LIBDIRS		:= -L ../build
LIBS 		:= -levo -lm

all: $(TEST_TRG)

%: %.c
	@echo [CC] $<
	$(CC) $(INCDIRS) $(LIBDIRS) -o $@ $< $(LIBS) -static

run: $(TEST_TRG)
	@for target in $(TEST_TRG); do \
		echo "[TEST] $$target"; \
		./$$target; \
	done

clean:
	@rm -f $(TEST_TRG)