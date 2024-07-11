
NAME				:= pyevo
PYBIND11_SRCS		:= $(NAME).cpp
PYBIND11_INCS 		:= $(shell python3 -m pybind11 --includes)
PYBIND11_DIR		:= .
PYBIND11_TRG		:= $(PYBIND11_DIR)/$(NAME)$(shell python3-config --extension-suffix)

$(shell mkdir -p $(PYBIND11_DIR))

all: $(PYBIND11_SRCS)
	@echo [CXX] $<
	@cc -O3 -Wall -shared -std=c++11 -fPIC $(PYBIND11_INCS) $(PYBIND11_SRCS) -o $(PYBIND11_TRG)
	@echo [DYN] $(PYBIND11_TRG)

test:
	@python3 demo.py

clean:
	@rm -f $(PYBIND11_TRG)