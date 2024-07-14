CCFLAGC := 
SIMFLAG := -cc -trace --timing -exe --build
INCDIRC := -I ./vsrc
LIBDIRC := 
LDFLAGS :=
SRCC 	:= $(filter-out edb.c, $(wildcard *.c))
CSRC 	:= $(shell find ./csrc -name "*.c" -or -name "*.cpp" -or -name "*.cc")
VSRC 	:= $(shell find ./vsrc -name "*.v")
LIBC 	:=

all: build

# edb-cl: $(SRCC)
# 	@echo [CC] $<
# 	@$(CC) -o $@ $(SRCC) $(CCFLAGC) $(INCDIRC) $(LIBDIRC) $(LIBC)

build: $(VSRC)
	@verilator $(SIMFLAG) \
	--top-module top \
	$(CSRC) \
	$(VSRC)

sim: build
	@./obj_dir/Vtop

clean:
	@rm -rf obj_dir
