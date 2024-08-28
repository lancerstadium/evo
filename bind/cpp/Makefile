CXX        := g++
SRCS       := $(wildcard ./*.cpp)
INCDIRS    := -I . -I ../../include
LIBDIRS    := -L ../../build
LIBS       := -levo -lm
TARGET     := demo

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRCS)
	@echo [CXX] Linking $@
	@$(CXX) -o $@ $^ $(INCDIRS) $(LIBDIRS) $(LIBS)

run: $(TARGET)
	@./$(TARGET)

clean:
	@rm -f $(TARGET)
