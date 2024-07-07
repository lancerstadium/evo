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
OPTIONS		:= -DSOB_APP_OFF
ASFLAGS		:= -g -ggdb -Wall -O3 $(OPTIONS)
CFLAGS		:= -g -ggdb -Wall -O3 $(OPTIONS)
CXXFLAGS	:= -g -ggdb -Wall -O3 $(OPTIONS)
INCDIRS		:= -I .
SRCDIRS		:= $(NAME) $(NAME)/** $(NAME)/**/**

SFILES		:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.S))
CFILES		:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.c))
CPPFILES	:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.cpp))

SDEPS		:= $(patsubst %, %, $(SFILES:.S=.o.d))
CDEPS		:= $(patsubst %, %, $(CFILES:.c=.o.d))
CPPDEPS		:= $(patsubst %, %, $(CPPFILES:.cpp=.o.d))
DEPS		:= $(SDEPS) $(CDEPS) $(CPPDEPS)

SOBJS		:= $(patsubst %, %, $(SFILES:.S=.o))
COBJS		:= $(patsubst %, %, $(CFILES:.c=.o))
CPPOBJS		:= $(patsubst %, %, $(CPPFILES:.cpp=.o)) 
OBJS		:= $(SOBJS) $(COBJS) $(CPPOBJS)

TRGDIR 		:= build
LIBTRG		:= $(TRGDIR)/lib$(NAME).a

CUR_TIME 	:= $(shell date +"%Y-%m-%d %H:%M:%S")


$(shell mkdir -p $(TRGDIR))

.PHONY: all clean test

all : $(LIBTRG)
	@$(MAKE) -s -C tests all

$(LIBTRG) : $(OBJS)
	@echo [AR] Archiving $@
	@$(AR) -rcs $@ $(OBJS)

$(SOBJS) : %.o : %.S
	@echo [AS] $<
	@$(AS) $(ASFLAGS) -MD -MP -MF $@.d $(INCDIRS) -c $< -o $@

$(COBJS) : %.o : %.c
	@echo [CC] $<
	@$(CC) $(CFLAGS) -MD -MP -MF $@.d $(INCDIRS) -c $< -o $@

$(CPPOBJS) : %.o : %.cpp
	@echo [CXX] $<
	@$(CXX) $(CXXFLAGS) -MD -MP -MF $@.d $(INCDIRS) -c $< -o $@

test:
	@$(MAKE) -s -C tests run

line:
	@wc -l `find ./ -name "*.c";find -name "*.h"`

commit:
	git add .
	git commit -m '$(CUR_TIME)'
	git push

clean:
	@$(RM) $(DEPS) $(OBJS) $(LIBTRG)
	@$(MAKE) -s -C tests clean

sinclude $(DEPS)
