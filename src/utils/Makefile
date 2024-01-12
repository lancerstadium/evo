
# 编译本文件夹文件

# C编译选项
CFLAGS = --std=c11 -g -static
# C头文件目录
INC_DIR = -I.

SRC_FILES = $(wildcard *.c)


all:
	ar rc libutils.a $(SRC_FILES)

test:
	cc $(CFLAGS) $(INC_DIR) $(SRC_FILES) -o utils-test
	./utils-test

clean:
	rm -f utils

