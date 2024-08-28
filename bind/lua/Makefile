CXX = g++
CXXFLAGS = -std=c++11 -fPIC
LDFLAGS = -shared
LUABIND_INCS = -I. -I../../include -I./LuaBridge
LIBDIRS = -L/usr/local/lib/lua -L../../build
LIBS = -lstdc++ -levo -llua -lm 

LUABIND_SRCS = luaevo.cpp
LUABIND_TRG = luaevo.so

all: $(LUABIND_TRG)

$(LUABIND_TRG): $(LUABIND_SRCS)
	@echo [CXX] $<
	@$(CXX) $(LUABIND_SRCS) $(LUABIND_INCS) $(LIBDIRS) $(CXXFLAGS) $(LDFLAGS) $(LIBS) -o $(LUABIND_TRG)
	@echo [DYN] $(LUABIND_TRG)

run:
	@lua demo.lua

clean:
	@rm -f $(LUABIND_TRG)