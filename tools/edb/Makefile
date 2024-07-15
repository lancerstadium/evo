CCFLAGS := -DEDB_DIFF_SERVER
INCDIRS := -I . -I ../../evo
LIBDIRS := -L ../../build
SRCS 	:= $(wildcard src/*.c)
LIBS 	:= -levo -lm
TRGDIR	:= build
SERTRG	:= $(TRGDIR)/edb

CCFLAGC := -g -ggdb -Wall -O3 -fPIC
INCDIRC := -I .
LIBDIRC := 
SRCC 	:= $(filter-out src/edb.c, $(wildcard src/*.c))
OBJC	:= $(SRCC:src/%.c=$(TRGDIR)/%.o)
OBJDC	:= $(SRCC:src/%.c=$(TRGDIR)/%.o.d)
LIBC 	:=
CLTTRG  := $(TRGDIR)/libedb.a

NPCDIR	:= npc

$(shell mkdir -p $(TRGDIR))

all: server client

server: $(SERTRG) 

client: $(CLTTRG)

$(SERTRG): $(SRCS)
	@echo [CC] $<
	@$(CC) -o $@ $(SRCS) $(CCFLAGS) $(INCDIRS) $(LIBDIRS) $(LIBS)

$(OBJC): $(SRCC)
	@echo [CC] $<
	@$(CC) -o $@ -c $< $(CCFLAGC) -MD -MP -MF $@.d $(INCDIRC) $(LIBDIRC) $(LIBC)

$(CLTTRG): $(OBJC)
	@echo [AR] Archiving $@
	@ar -rcs $@ $^

run: $(SERTRG)
	@./$(SERTRG)

sim: $(CLTTRG)
	@$(MAKE) -s -C $(NPCDIR) sim

clean:
	@rm -rf $(TRGDIR)
	@$(MAKE) -s -C $(NPCDIR) clean
