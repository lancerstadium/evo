# Makefile for library

CROSS_COMPILE	?= 

AS			:= $(CROSS_COMPILE)gcc -x assembler-with-cpp
CC			:= $(CROSS_COMPILE)gcc
CXX			:= $(CROSS_COMPILE)g++
LD			:= $(CROSS_COMPILE)ld
AR			:= $(CROSS_COMPILE)ar
OC			:= $(CROSS_COMPILE)objcopy
OD			:= $(CROSS_COMPILE)objdump
RM			:= rm -fr

NAME		:= evo
LIBS		:= -l flatccrt
LIBDIRS     := -L ./sub/flatcc/lib
INCDIRS		:= -I . -I sub/flatcc/include
SRCDIRS		:= $(NAME) $(NAME)/** $(NAME)/**/**

ARCH_DEP	?=
ifeq ($(CROSS_COMPILE),riscv64-linux-gnu-)
ARCH_DEP	:= 
else ifeq ($(CROSS_COMPILE), aarch64-linux-gnu-)
ARCH_DEP	:= 
endif

NOWARNS		:= -Wno-misleading-indentation -Wno-unused-result
OPTIONS		:= $(LIBS) $(LIBDIRS) $(NOWARNS) $(ARCH_DEP)
ASFLAGS		:= -g -ggdb -Wall -O3 $(OPTIONS) -fPIC
CFLAGS		:= -g -ggdb -Wall -O3 $(OPTIONS) -fPIC
CXXFLAGS	:= -g -ggdb -Wall -O3 $(OPTIONS) -fPIC
TRGDIR 		:= build
OBJDIR		:= $(TRGDIR)/obj
LIBTRG		:= $(TRGDIR)/lib$(NAME).a

SFILES		:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.S))
CFILES		:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.c))
CPPFILES	:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.cpp))

SDEPS       := $(patsubst %, %, $(SFILES:$(NAME)/%.S=$(OBJDIR)/%.o.d))
CDEPS       := $(patsubst %, %, $(CFILES:$(NAME)/%.c=$(OBJDIR)/%.o.d))
CPPDEPS     := $(patsubst %, %, $(CPPFILES:$(NAME)/%.cpp=$(OBJDIR)/%.o.d))
DEPS        := $(SDEPS) $(CDEPS) $(CPPDEPS)

SOBJS       := $(patsubst %, %, $(SFILES:$(NAME)/%.S=$(OBJDIR)/%.o))
COBJS       := $(patsubst %, %, $(CFILES:$(NAME)/%.c=$(OBJDIR)/%.o))
CPPOBJS     := $(patsubst %, %, $(CPPFILES:$(NAME)/%.cpp=$(OBJDIR)/%.o)) 
OBJS        := $(SOBJS) $(COBJS) $(CPPOBJS)

CUR_TIME 	:= $(shell date +"%Y-%m-%d %H:%M:%S")


$(shell mkdir -p $(TRGDIR) $(TRGDIR)/obj)

.PHONY: all clean test line tool

all : $(LIBTRG)
	@$(MAKE) -s -C tests all

$(LIBTRG) : $(OBJS)
	@echo [AR] Archiving $@
	@$(AR) -rcs $@ $(OBJS)

$(OBJDIR)/%.o : $(NAME)/%.S
	@echo [AS] $<
	@mkdir -p $(dir $@)
	@$(AS) $(ASFLAGS) -MD -MP -MF $@.d $(INCDIRS) -c $< -o $@

$(OBJDIR)/%.o : $(NAME)/%.c
	@echo [CC] $<
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) -MD -MP -MF $@.d $(INCDIRS) -c $< -o $@

$(OBJDIR)/%.o : $(NAME)/%.cpp
	@echo [CXX] $<
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) -MD -MP -MF $@.d $(INCDIRS) -c $< -o $@

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
	git commit -m '$(CUR_TIME)'
	git push

clean:
	@$(RM) $(DEPS) $(OBJS) $(LIBTRG)
	@$(RM) $(OBJDIR)
	@$(MAKE) -s -C tests clean
	@$(MAKE) -s -C tools clean

sinclude $(DEPS)