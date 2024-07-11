
CXX					:= g++
NAME				:= pyevo
PYBIND_SRCS			:= $(NAME).cpp
PYBIND_INCS 		:= $(shell python3 -m pybind11 --includes)
PYBIND_DIR			:= .
PYBIND_TRG			:= $(PYBIND_DIR)/$(NAME)$(shell python3-config --extension-suffix)

$(shell mkdir -p $(PYBIND_DIR))

all: $(PYBIND_SRCS)
	@echo [CXX] $<
	@$(CXX) -O3 -Wall -shared -std=c++11 -fPIC $(PYBIND_INCS) $(PYBIND_SRCS) -o $(PYBIND_TRG)
	@echo [DYN] $(PYBIND_TRG)

test:
	@python3 demo.py

clean:
	@rm -f $(PYBIND_TRG)