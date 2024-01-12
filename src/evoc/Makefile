# C编译选项
CFLAGS = --std=c11 -g -static
# C头文件目录
INC_DIR = -I. -I../utils

SRC_FILES = evoc.c
SRC_FILES += ../utils/log.c

all:
	cc $(CFLAGS) $(INC_DIR) $(SRC_FILES) -o evoc 

run:
	./evoc 123 > tmp.s
	cc tmp.s -o tmp
	./tmp && echo $?

test:
	./evoc-test.sh
	

testa:
	bash -x ./evoc-test.sh

clean:
	rm -f evoc *.s *.o tmp*

.PHONY: run test testa clean