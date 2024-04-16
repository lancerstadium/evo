CC = gcc
CFLAGS = --std=gnu11 -g -static
CODEDIR = src
BINDIR = bin
OBJDIR = obj
LDFLAGS = $(foreach dir, $(FOLDERS), -L$(CODEDIR)/$(dir))
INCDIR = $(foreach dir, $(FOLDERS), -I$(CODEDIR)/$(dir))
EVO = $(CODEDIR)/app/evo
EMU = $(CODEDIR)/app/emu
TEST_UTIL = $(CODEDIR)/app/test-util
TEST_DEC = $(CODEDIR)/app/test-dec
EXES = $(EVO).c $(EMU).c $(TEST_UTIL).c $(TEST_DEC).c
SRCS = $(filter-out $(EXES), $(foreach dir, $(FOLDERS), $(wildcard $(CODEDIR)/$(dir)/*.c)))
OBJS = $(patsubst $(CODEDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

FOLDERS = app core dec emu fmt jit mem util

# 创建目录
$(shell mkdir -p $(OBJDIR) $(foreach dir, $(FOLDERS), $(OBJDIR)/$(dir)))

all: evo emu test-util test-dec

evo: $(OBJS) $(OBJDIR)/evo.o
	$(CC) $(LDFLAGS) $^ -o $(BINDIR)/evo

emu: $(OBJS) $(OBJDIR)/emu.o
	$(CC) $(LDFLAGS) $^ -o $(BINDIR)/emu

test-util: $(OBJS) $(OBJDIR)/test-util.o
	$(CC) $(LDFLAGS) $^ -o $(BINDIR)/test-util

test-dec: $(OBJS) $(OBJDIR)/test-dec.o
	$(CC) $(LDFLAGS) $^ -o $(BINDIR)/test-dec

$(OBJDIR)/%.o: $(CODEDIR)/%.c
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@

$(OBJDIR)/evo.o: $(EVO).c
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@

$(OBJDIR)/emu.o: $(EMU).c
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@

$(OBJDIR)/test-util.o: $(TEST_UTIL).c
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@

$(OBJDIR)/test-dec.o: $(TEST_DEC).c
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@

run:


test:
	$(BINDIR)/util-test

clean:
	rm -f $(BINDIR)/evo $(BINDIR)/emu $(BINDIR)/test-util $(BINDIR)/test-dec
	rm -f $(OBJDIR)/*.o $(foreach dir, $(FOLDERS), $(OBJDIR)/$(dir)/*.o)

.PHONY: evo util-test run test clean
