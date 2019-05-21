# Makefile to create distlearn library
# created first in pydistlearn
# ToraNova 2019

# submodules list
gsub := pyplasma

PYTHV ?= python3.6

#default build target
.phony: all
all: submod 

.phony: debug
debug: $(gsub)
	git submodule update --remote $<
	cd $< && $(MAKE) PYTHV=$(PYTHV) -C debug

.phony: submod
submod: $(gsub)
	git submodule update --remote $<
	cd $< && $(MAKE) PYTHV=$(PYTHV)

$(gsub):
	git submodule init

.phony: clean distclean test
clean: $(gsub)
	cd $(gsub) && $(MAKE) clean

distclean: $(gsub)
	rm -rf pyplasma	

test:
	python3 plasma_test.py
