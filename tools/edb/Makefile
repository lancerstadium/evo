CCFLAGS := -DEDB_DIFF_SERVER
INCDIRS := -I . -I ../../evo
LIBDIRS := -L ../../build
SRCS 	:= $(filter-out edb-cl.c, $(wildcard *.c))
LIBS 	:= -levo -lm

all: edb

edb: $(SRCS)
	@echo [CC] $<
	@$(CC) -o $@ $(SRCS) $(CCFLAGS) $(INCDIRS) $(LIBDIRS) $(LIBS)

clean:
	@rm -f edb edb-cl
