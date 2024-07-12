.PHONY: all run clean

CXX                 := g++
NAME                := pyevo
PYBIND_SRCS         := $(NAME).cpp
PYBIND_INCS         := -I . -I ../../evo $(shell python3 -m pybind11 --includes)
LIBDIRS             := -L ../../build
LIBS                := -levo -lm
PYBIND_DIR          := .
PYBIND_TRG          := $(PYBIND_DIR)/$(NAME)$(shell python3-config --extension-suffix)

$(shell mkdir -p $(PYBIND_DIR))

all: $(PYBIND_TRG)

$(PYBIND_TRG): $(PYBIND_SRCS)
	@echo [CXX] $<
	@$(CXX) -O3 -Wall -shared -std=c++11 -fPIC $(PYBIND_INCS) $(LIBDIRS) $(PYBIND_SRCS) $(LIBS) -o $(PYBIND_TRG)
	@echo [DYN] $(PYBIND_TRG)

check:
	ldd -r $(PYBIND_TRG)

run:
	@python3 demo.py

clean:
	@rm -f $(PYBIND_TRG)
