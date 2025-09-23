CSRC 	:= $(shell find ./csrc -name "*.c" -or -name "*.cpp" -or -name "*.cc")
VSRC 	:= $(shell find ./vsrc -name "*.v")
SIMFLAG := -cc -trace --timing -exe --build
LDFLAGS := "-ledb -L../../build/"

all: build

build: $(VSRC)
	@verilator $(SIMFLAG) \
	--top-module top \
	$(CSRC) \
	$(VSRC) \
	-LDFLAGS $(LDFLAGS) 

sim: build
	@./obj_dir/Vtop

clean:
	@rm -rf obj_dir
	@rm -rf *.vcd
