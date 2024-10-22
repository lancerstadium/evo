# Makefile for library

# System Build Tool
CROSS_COMPILE	?= 

# compile tool
AS			:= $(CROSS_COMPILE)gcc -x assembler-with-cpp
CC			:= $(CROSS_COMPILE)gcc
CXX			:= $(CROSS_COMPILE)g++
LD			:= $(CROSS_COMPILE)ld
AR			:= $(CROSS_COMPILE)ar
OC			:= $(CROSS_COMPILE)objcopy
OD			:= $(CROSS_COMPILE)objdump
RM			:= rm -fr
NVCC		:= nvcc

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

DEVICE      := cpu
# File & Dependence
LIBS		:= 
LIBDIRS     := 
INCDIRS		:= -I./include
SRCDIRS		:= src src/** src/**/**

ARCH_DEP		?=
ifeq ($(CROSS_COMPILE),riscv64-linux-gnu-)
	ARCH_DEP	:= 
else ifeq ($(CROSS_COMPILE), aarch64-linux-gnu-)
	ARCH_DEP	:= 
endif

# Options
DEV_ENB		:=				# Compile with develop
GUI_ENB		:= 				# Compile with evo-gui
TRAIN_ENB	:=				# Compile with train-mode
ACC_ENB 	:=				# Compile with acclerator

# Options: DEV
DEV_DEP		?=
ifneq ($(GUI_ENB),)
	DEV_DEP		:= -g -ggdb
endif

# Options: GUI
GUI_DEP			?=
ifneq ($(GUI_ENB),)
	ifeq ($(PLATFORM),Linux)
		GUI_DEP		:= -lX11
	else ifeq ($(PLATFORM),Darwin)
		GUI_DEP		:= -framework Cocoa
	else ifeq ($(PLATFORM),Windows)
		GUI_DEP		:=
	endif
endif

# Options: TRAIN
TRAIN_DEP		:=
ifneq ($(TRAIN_ENB),)
	TRAIN_DEP		:=
endif

# Options: ACC
COMMA			:=,
ACC_ENB_LIST 	:= $(subst $(COMMA), ,$(ACC_ENB))
ACC_ENB_RES		:=
ACC_ENB_CUDA	:=
ACC_ENB_METAL	:=
ACC_DEP 		:=
ACC_DEP_CUDA	:=
ACC_DEP_METAL	:=
ifneq ($(ACC_ENB),)
define set_acc_dep
	ifeq ($(strip $(1)),cuda)
		ACC_DEP_CUDA  	+= -gencode arch=compute_50,code=sm_50 \
						-gencode arch=compute_52,code=sm_52 \
						-gencode arch=compute_86,code=sm_86
		ACC_ENB_RES		+= cuda 
		ACC_ENB_CUDA 	:= 1
	else ifeq ($(strip $(1)),metal)
		ACC_DEP_METAL 	+= 
		ACC_ENB_RES		+= metal
		ACC_ENB_METAL	:= 1
	endif
endef
$(foreach arch,$(ACC_ENB_LIST),$(eval $(call set_acc_dep,$(arch))))
endif


NOWARNS		:= -Wno-misleading-indentation -Wno-unused-result
OPTIONS		:= $(LIBDIRS) $(LIBS) $(DEV_ENB) $(ARCH_DEP) $(ACC_DEP) $(GUI_DEP) $(TRAIN_DEP) $(NOWARNS)
ASFLAGS		:= $(OPTIONS) -Wall -O3 -fPIC
CFLAGS		:= $(OPTIONS) -Wall -O3 -fPIC
CXXFLAGS	:= $(OPTIONS) -Wall -O3 -fPIC
TRGDIR 		:= build
OBJDIR		:= $(TRGDIR)/obj
LIBTRG		:= $(TRGDIR)/lib$(NAME).a

SFILES		:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.S))
CFILES		:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.c))
CUFILES		:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.cu))
CPPFILES	:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.cpp))
CFGFILE		:= include/evo/config.h

SDEPS       := $(patsubst %, %, $(SFILES:src/%.S=$(OBJDIR)/%.o.d))
CDEPS       := $(patsubst %, %, $(CFILES:src/%.c=$(OBJDIR)/%.o.d))
CUDEPS     	:= $(patsubst %, %, $(CUFILES:src/%.cu=$(OBJDIR)/%.o.d))
CPPDEPS     := $(patsubst %, %, $(CPPFILES:src/%.cpp=$(OBJDIR)/%.o.d))
DEPS        := $(SDEPS) $(CDEPS) $(CPPDEPS)

SOBJS       := $(patsubst %, %, $(SFILES:src/%.S=$(OBJDIR)/%.o))
COBJS       := $(patsubst %, %, $(CFILES:src/%.c=$(OBJDIR)/%.o))
CUOBJS		:= $(patsubst %, %, $(CUFILES:src/%.cu=$(OBJDIR)/%.o))
CPPOBJS     := $(patsubst %, %, $(CPPFILES:src/%.cpp=$(OBJDIR)/%.o)) 
OBJS 		:= $(SOBJS) $(COBJS) $(CPPOBJS)
# Check config if defined EVO_ACC_CUDA
ifneq (,$(findstring EVO_ACC_CUDA,$(shell cat $(CFGFILE))))
    OBJS 	+= $(CUOBJS)
endif

CURTIME 	:= $(shell date +"%Y-%m-%d %H:%M:%S")


.PHONY: all config clean test release line tool

all : $(LIBTRG)
	@$(MAKE) -s -C tests all

config: $(CFGFILE)
	@echo "[Config] >> \t\t\e[33;1m$(CFGFILE)\e[0m"
	@echo "  - item:   \t\t$(NAME)"
	@echo "/* Auto-generated config.h */" > $(CFGFILE)
	@echo "#define EVO_VERSION \"$(VERSION)\"" >> $(CFGFILE)
	@echo "  - version: \t\t$(VERSION)"
	@echo "#define EVO_PLATFORM \"$(PLATFORM)\"" >> $(CFGFILE)
	@echo "  - platform: \t\t$(PLATFORM)"
	@echo "#define EVO_DEVICE \"$(DEVICE)\"" >> $(CFGFILE)
	@echo "  - device: \t\t$(DEVICE)"
	@echo "[Option]"
ifneq ($(GUI_ENB),)
	@echo "#define EVO_GUI_ENB" >> $(CFGFILE)
	@echo "  - gui:    \t\t\e[32;1m$(GUI_DEP)\e[0m"
else
	@echo "  - gui:    \t\t\e[31;1moff\e[0m"
endif
ifneq ($(TRAIN_ENB),)
	@echo "#define EVO_TRAIN_ENB" >> $(CFGFILE)
	@echo "  - train:  \t\t\e[32;1mon\e[0m"
else
	@echo "  - train:  \t\t\e[31;1moff\e[0m"
endif
ifneq ($(ACC_ENB_RES),)
ifeq ($(ACC_ENB_CUDA),1)
	@echo "#define EVO_ACC_CUDA" >> $(CFGFILE)
endif
ifeq ($(ACC_ENB_METAL),1)
	@echo "#define EVO_ACC_X86_64" >> $(CFGFILE)
endif
	@echo "  - acc:    \t\t\e[32;1m$(ACC_ENB_RES)\e[0m"
else
	@echo "  - acc:    \t\t\e[31;1moff\e[0m"
endif


$(CFGFILE):
	@echo "Creating $(CFGFILE)..."
	@touch $(CFGFILE)

$(LIBTRG) : $(OBJS)
	$(shell mkdir -p $(TRGDIR) $(TRGDIR)/obj)
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

$(OBJDIR)/%.o : src/%.cu
	@echo [CU] $<
	@mkdir -p $(dir $@)
	@$(NVCC) $(ACC_DEP_CUDA) --compiler-options "-MD -MP -MF $@.d $(INCDIRS) $(CFLAGS)" -c $< -o $@

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


release:
	@mkdir -p $(TRGDIR)/evo-release-$(VERSION)/
	@cp -r src/ $(TRGDIR)/evo-release-$(VERSION)/
	@cp -r include/ $(TRGDIR)/evo-release-$(VERSION)/
	@cp -r Makefile $(TRGDIR)/evo-release-$(VERSION)/
	@tar -zcvf $(TRGDIR)/evo-release-$(VERSION).tar.gz $(TRGDIR)/evo-release-$(VERSION)/

clean:
	@$(RM) $(DEPS) $(OBJS) $(LIBTRG)
	@$(RM) $(OBJDIR)
	@$(MAKE) -s -C tests clean
	@$(MAKE) -s -C tools clean

sinclude $(DEPS)