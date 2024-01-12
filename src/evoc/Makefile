CFLAGS = c11 -g -static		# C编译选项

all:
	cc evoc.c -o evoc

run:
	./evoc 123 > tmp.s
	cc tmp.s -o tmp
	./tmp
	echo $?

test:
	bash -x ./evoc-test.sh

testq:
	./evoc-test.sh

clean:
	rm -f evoc *.s tmp