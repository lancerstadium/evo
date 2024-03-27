# C编译选项
CC = gcc
CFLAGS = --std=c11 -g -static
LDFLAGS = -L./app -L./core -L./util
# C头文件目录
INCDIR = -I./app -I./core -I./util
SRCS=$(wildcard ./app/*.c ./core/*.c ./util/*.c)
EXES=./util/util-test.c
OBJS=$(patsubst %.c, %.o, $(filter-out $(EXES), $(SRCS)))


all: $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o evo

$(OBJS): %.o: %.c
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@


run:


clean:
	rm -f evo ./app/*.o ./util/*.o ./core/*.o

.PHONY: run clean