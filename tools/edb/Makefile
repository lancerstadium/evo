INCDIRS := -I . -I ../../evo
LIBDIRS := -L ../../build
LIBS := -levo -lm

all: edb

edb: *.c *.h
	@echo [CC] $<
	@$(CC) -o edb *.c $(INCDIRS) $(LIBDIRS) $(LIBS)

clean:
	@rm -f edb
