# Makefile for library

# System Build Tool
CROSS_COMPILE	?= 

AS			:= $(CROSS_COMPILE)gcc -x assembler-with-cpp
CC			:= $(CROSS_COMPILE)gcc
CXX			:= $(CROSS_COMPILE)g++
LD			:= $(CROSS_COMPILE)ld
AR			:= $(CROSS_COMPILE)ar
OC			:= $(CROSS_COMPILE)objcopy
OD			:= $(CROSS_COMPILE)objdump
RM			:= rm -fr

# Application Information
NAME		:= evo
VERSION 	:= 1.0.0
AUTHOR		:= lancerstadium

# Platform Information
PLATFORM  	?=
ifeq ($(shell uname),Linux)
    PLATFORM := Linux
else ifeq ($(shell uname),Darwin)
    PLATFORM := macOS
else ifeq ($(OS),Windows)
    PLATFORM = Windows
endif

# File & Dependence
LIBS		:= 
LIBDIRS     := 
INCDIRS		:= -I./include
SRCDIRS		:= src src/** src/**/**

ARCH_DEP	?=
ifeq ($(CROSS_COMPILE),riscv64-linux-gnu-)
	ARCH_DEP	:= 
else ifeq ($(CROSS_COMPILE), aarch64-linux-gnu-)
	ARCH_DEP	:= 
endif

# Options
GUI_ENB		:= 			# Compile with evo-gui

# Options: GUI
GUI_DEP		?=
ifneq ($(GUI_ENB),)
	ifeq ($(PLATFORM),Linux)
		GUI_DEP		:= -lX11
	else ifeq ($(PLATFORM),Darwin)
		GUI_DEP		:= -framework Cocoa
	else ifeq ($(PLATFORM),Windows)
		GUI_DEP		:=
	endif
endif

NOWARNS		:= -Wno-misleading-indentation -Wno-unused-result
OPTIONS		:= $(LIBDIRS) $(LIBS) $(ARCH_DEP) $(GUI_DEP) $(NOWARNS)
ASFLAGS		:= $(OPTIONS) -g -ggdb -Wall -O3 -fPIC
CFLAGS		:= $(OPTIONS) -g -ggdb -Wall -O3 -fPIC
CXXFLAGS	:= $(OPTIONS) -g -ggdb -Wall -O3 -fPIC
TRGDIR 		:= build
OBJDIR		:= $(TRGDIR)/obj
LIBTRG		:= $(TRGDIR)/lib$(NAME).a

SFILES		:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.S))
CFILES		:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.c))
CPPFILES	:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.cpp))
CFGFILE		:= include/config.h

SDEPS       := $(patsubst %, %, $(SFILES:src/%.S=$(OBJDIR)/%.o.d))
CDEPS       := $(patsubst %, %, $(CFILES:src/%.c=$(OBJDIR)/%.o.d))
CPPDEPS     := $(patsubst %, %, $(CPPFILES:src/%.cpp=$(OBJDIR)/%.o.d))
DEPS        := $(SDEPS) $(CDEPS) $(CPPDEPS)

SOBJS       := $(patsubst %, %, $(SFILES:src/%.S=$(OBJDIR)/%.o))
COBJS       := $(patsubst %, %, $(CFILES:src/%.c=$(OBJDIR)/%.o))
CPPOBJS     := $(patsubst %, %, $(CPPFILES:src/%.cpp=$(OBJDIR)/%.o)) 
OBJS        := $(SOBJS) $(COBJS) $(CPPOBJS)

CURTIME 	:= $(shell date +"%Y-%m-%d %H:%M:%S")


$(shell mkdir -p $(TRGDIR) $(TRGDIR)/obj)

.PHONY: all config clean test line tool

all : $(LIBTRG)
	@$(MAKE) -s -C tests all

config: $(CFGFILE)
	@echo "[Config] >> \t\t\e[33;1m$(CFGFILE)\e[0m"
	@echo "application: \t\t$(NAME)"
	@echo "/* Auto-generated config.h */" > $(CFGFILE)
	@echo "#define EVO_VERSION \"$(VERSION)\"" >> $(CFGFILE)
	@echo "version: \t\t$(VERSION)"
	@echo "#define EVO_PLATFORM \"$(PLATFORM)\"" >> $(CFGFILE)
	@echo "platform: \t\t$(PLATFORM)"
ifneq ($(GUI_ENB),)
	@echo "#define EVO_GUI_ENB" >> $(CFGFILE)
	@echo "option-gui: \t\t\e[32;1mon\e[0m"
else
	@echo "option-gui: \t\t\e[31;1moff\e[0m"
endif

$(LIBTRG) : $(OBJS)
	@echo [AR] Archiving $@
	@$(AR) -rcs $@ $(OBJS)

$(OBJDIR)/%.o : src/%.S
	@echo [AS] $<
	@mkdir -p $(dir $@)
	@$(AS) -MD -MP -MF $@.d $(INCDIRS) -c $< -o $@ $(ASFLAGS)

$(OBJDIR)/%.o : src/%.c
	@echo [CC] $<
	@mkdir -p $(dir $@)
	@$(CC) -MD -MP -MF $@.d $(INCDIRS) -c $< -o $@ $(CFLAGS)

$(OBJDIR)/%.o : src/%.cpp
	@echo [CXX] $<
	@mkdir -p $(dir $@)
	@$(CXX) -MD -MP -MF $@.d $(INCDIRS) -c $< -o $@ $(CXXFLAGS)

test: $(LIBTRG)
	@$(MAKE) -s -C tests run

tool: $(LIBTRG)
	@$(MAKE) -s -C tools all

line:
	@wc -l `find ./ -name "*.c";find -name "*.h"`

head:
	@grep -h '#include' $(SRCDIRS) | awk '{print $2}' | sort | uniq -c

commit:
	git add .
	git commit -m '$(CURTIME)'
	git push

clean:
	@$(RM) $(DEPS) $(OBJS) $(LIBTRG)
	@$(RM) $(OBJDIR)
	@$(MAKE) -s -C tests clean
	@$(MAKE) -s -C tools clean

sinclude $(DEPS)