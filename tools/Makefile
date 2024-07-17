SUBDIRS := $(wildcard */)
EXCLUDED_DIRS := bin build
SUBDIRS := $(filter-out $(EXCLUDED_DIRS), $(SUBDIRS))

ifeq (,$(strip $(SUBDIRS)))
$(error No subdirectories found)
endif

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	@if [ -f $@/Makefile ]; then \
		$(MAKE) -s -C $@; \
	fi


clean:
	@for dir in $(SUBDIRS); do \
		if [ -f $$dir/Makefile ]; then \
			$(MAKE) -s -C $$dir clean; \
		fi \
	done
