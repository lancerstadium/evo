
# 编译本文件夹文件

# C编译选项
CFLAGS = --std=c11 -g -static
# C头文件目录
INC_DIR = -I.

SRC_FILES = $(wildcard *.c)


all:
	ar rc libutil.a $(SRC_FILES)

test:
	cc $(CFLAGS) $(INC_DIR) $(SRC_FILES) -o util-test
	./util-test

clean:
	rm -f libutil.a *.o util-test

