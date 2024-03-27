# C编译选项
CC = gcc
CFLAGS = --std=c11 -g -static
LDFLAGS = -L../core -L../util
# C头文件目录
INCDIR = -I. -I../core -I../util
SRCS=$(wildcard *.c ../core/*.c ../util/*.c)
# OBJS=$(SRCS:.c=.o)
EXES=../util/util-test.c
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
	rm -f evoc *.o tmp tmp.s ../util/*.o ../core/*.o

.PHONY: run test testa clean