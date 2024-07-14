SUBDIRS := $(wildcard */)
EXCLUDED_DIRS := bin build
SUBDIRS := $(filter-out $(EXCLUDED_DIRS), $(SUBDIRS))

ifeq (,$(strip $(SUBDIRS)))
$(error No subdirectories found)
endif

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	@$(MAKE) -s -C $@


clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -s -C $$dir clean; \
	done
