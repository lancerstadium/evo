CC = gcc
APP = evo
INCDIR = src
SRCDIR = src
BINDIR = bin
OBJDIR = obj
SOBDIR = src/sob
INCFLAGS = $(addprefix -I,$(INCDIR))
OPTFLAGS = 
CFLAGS = -g -O2 -Wall -Wextra $(INCFLAGS) -rdynamic -DSOB_APP_OFF -DSOB_LOG_DBG_OFF $(OPTFLAGS)
PREFIX ?= /usr/local
LIBS = -ldl $(OPTLIBS)
OS=$(shell lsb_release -si)
ifeq ($(OS),Ubuntu)
	LDLIBS=-l$(APP) -L./build -lm
endif

SRCS=$(wildcard src/**/*.c src/*.c)
OBJS=$(patsubst %.c,%.o,$(SRCS))

SRCTEST=$(wildcard test/*_tests.c)
TESTS=$(patsubst %.c,%,$(SRCTEST))

TARGET=build/lib$(APP).a
SO_TARGET=$(patsubst %.a,%.so,$(TARGET))
SOB_TARGET=sob
BIN_TARGET=bin/$(APP)

# Config
CFG_REPORT=0
CFG_TEST=

# The Target Build
all: $(SOB_TARGET) $(TARGET) tests

dev: CFLAGS=-g -Wall $(INCFLAGS) -Wall -Wextra $(OPTFLAGS)
dev: all

$(TARGET): CFLAGS += -fPIC
$(TARGET): build $(OBJS)
	ar rcs $@ $(OBJS)
	ranlib $@

# $(SO_TARGET): $(TARGET) $(OBJS)
# 	$(CC) -shared -o $@ $(OBJS)

$(SOB_TARGET): 
	$(CC) -o $@ $(SOBDIR)/sob.c -I$(SOBDIR)

# $(BIN_TARGET): build $(OBJS)
# 	$(CC) -o $@ $(OBJS)

build:
	@mkdir -p build
	@mkdir -p bin

# The Unit Tests
.PHONY: tests
tests: LDLIBS += $(TARGET)
tests: $(TESTS)
ifeq ($(CFG_REPORT),1)
	@echo "tests report: ./test/tests.report"
	@sh ./test/runtests.sh $(CFG_TEST) | sed 's/\x1B\[[0-9;]*[JKmsu]//g' > test/tests.report
else
	@sh ./test/runtests.sh $(CFG_TEST)
endif

test:
	@$(VALGRIND) ./test/$(SECOND_GOAL)_tests 2>> test/tests.log

valgrind:
	@echo "valgrind log: ./test/valgrind.log"
	VALGRIND="valgrind --log-file=./test/valgrind.log" $(MAKE)

log:
	@tail -n $$(($$(tac test/tests.log | grep -m 1 -n '^────── Run' | cut -d: -f1) + 1)) test/tests.log | sed '/^$$/d'

# The Cleaner
clean:
	rm -f $(SOB_TARGET)
	rm -rf build $(OBJS) $(TESTS)
	rm -f test/tests.log test/valgrind.log
	find . -name "*.gc*" -exec rm {} \;
	rm -rf `find . -name "*.dSYM" -print`

# The Install
install: all
	install -d $(DESTDIR)/$(PREFIX)/lib/
	install $(TARGET) $(DESTDIR)/$(PREFIX)/lib/

commit:
	git add .
	git commit -m "$(shell date)"
	git push

# The Checker
BADFUNCS='[^_.>a-zA-Z0-9](str(n?cpy|n?cat|xfrm|n?dup|str|pbrk|tok|_)|stpn?cpy|a?sn?printf|byte_)'
check:
	@echo Files with potentially dangerous functions.
	@egrep $(BADFUNCS) $(SRCS) || true
