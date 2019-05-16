# Makefile to create distlearn library
# created first in pydistlearn
# ToraNova 2019

# submodules list
gsub := pyplasma

#default build target
.phony: all
all: submod 

.phony: debug
debug: $(gsub)
	git submodule update --remote $<
	cd $< && $(MAKE) debug

.phony: submod
submod: $(gsub)
	git submodule update --remote $<
	cd $< && $(MAKE)

$(gsub):
	git submodule init

.phony: clean
clean:
	rm -rf pyplasma	
