CCFLAGC := 
INCDIRC := -I .
LIBDIRC := 
SRCC 	:= $(filter-out edb.c, $(wildcard *.c))
LIBC 	:=

all: edb-cl

edb-cl: $(SRCC)
	@echo [CC] $<
	@$(CC) -o $@ $(SRCC) $(CCFLAGC) $(INCDIRC) $(LIBDIRC) $(LIBC)

clean:
	@rm -f edb edb-cl
