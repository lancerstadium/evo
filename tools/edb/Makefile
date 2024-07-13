CCFLAGS := -DEDB_DIFF_SERVER
INCDIRS := -I . -I ../../evo
LIBDIRS := -L ../../build
EDBSRCS := $(filter-out edb-cl.c, $(wildcard *.c))
CLSRCS := $(filter-out edb.c, $(wildcard *.c))
LIBS := -levo -lm

all: edb edb-cl

edb: $(EDBSRCS)
	@echo [CC] $<
	@$(CC) -o $@ $(EDBSRCS) $(CCFLAGS) $(INCDIRS) $(LIBDIRS) $(LIBS)

edb-cl: $(CLSRCS)
	@echo [CC] $<
	@$(CC) -o $@ $(CLSRCS) $(CCFLAGS) $(INCDIRS) $(LIBDIRS) $(LIBS)

clean:
	@rm -f edb edb-cl
