CC = gcc
CFLAGS = --std=gnu11 -g -static
CODEDIR = src
BINDIR = bin
OBJDIR = obj
LDFLAGS = $(foreach dir, $(FOLDERS), -L$(CODEDIR)/$(dir))
INCDIR = $(foreach dir, $(FOLDERS), -I$(CODEDIR)/$(dir))
EVO = $(CODEDIR)/app/evo
UTIL_TEST = $(CODEDIR)/app/util-test
EXES = $(EVO).c $(UTIL_TEST).c
SRCS = $(filter-out $(EXES), $(foreach dir, $(FOLDERS), $(wildcard $(CODEDIR)/$(dir)/*.c)))
OBJS = $(patsubst $(CODEDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

FOLDERS = app core util mem fmt

# 创建目录
$(shell mkdir -p $(OBJDIR) $(foreach dir, $(FOLDERS), $(OBJDIR)/$(dir)))

all: evo util-test

evo: $(OBJS) $(OBJDIR)/evo.o
	$(CC) $(LDFLAGS) $^ -o $(BINDIR)/evo

util-test: $(OBJS) $(OBJDIR)/util-test.o
	$(CC) $(LDFLAGS) $^ -o $(BINDIR)/util-test

$(OBJDIR)/%.o: $(CODEDIR)/%.c
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@

$(OBJDIR)/evo.o: $(EVO).c
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@

$(OBJDIR)/util-test.o: $(UTIL_TEST).c
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@

run:


test:
	$(BINDIR)/util-test

clean:
	rm -f $(BINDIR)/evo $(BINDIR)/util-test
	rm -f $(foreach dir, $(FOLDERS), $(OBJDIR)/$(dir)/*.o)

.PHONY: evo util-test run test clean
