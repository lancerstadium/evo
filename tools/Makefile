.PHONY: all clean

TOOL_DIR := 

SRC_TOOL := $(wildcard $(TOOL_DIR)/*.c)
TOOL_TRG := $(patsubst $(TOOL_DIR)/%.c,$(TOOL_DIR)/%,$(SRC_TOOL))
INCDIRS		:= -I . -I ../evo
LIBDIRS		:= -L ../build
LIBS 		:= -levo -lm

all: $(TOOL_TRG)

$(TOOL_DIR)/%: $(TOOL_DIR)/%.c
	@echo [CC] $<
	@$(CC) $(INCDIRS) $(LIBDIRS) -o $@ $< $(LIBS) -static

clean:
	@rm -f $(TOOL_TRG)