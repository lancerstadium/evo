# C编译选项
CC = gcc
CFLAGS = --std=c11 -g -static
LDFLAGS = -L../utils
# C头文件目录
INCDIR = -I. -I../utils
SRCS=$(wildcard *.c ../utils/*.c)
# OBJS=$(SRCS:.c=.o)
EXES=../utils/utils-test.c
OBJS=$(patsubst %.c, %.o, $(filter-out $(EXES), $(SRCS)))


all: $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o evoc

$(OBJS): %.o: %.c
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@


run:
	./evoc 123 > tmp.s
	cc tmp.s -o tmp
	./tmp && echo $?

test:
	./evoc-test.sh
	

testa:
	bash -x ./evoc-test.sh

clean:
	rm -f evoc *.o tmp tmp.s ../utils/*.o

.PHONY: run test testa clean