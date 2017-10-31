ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif

# Default target, can be overriden by command line or environment
RTE_TARGET ?= x86_64-native-linuxapp-gcc

include $(RTE_SDK)/mk/rte.vars.mk

# binary name
APP = benchmark

# all source are stored in SRCS-y
SRCS-y := benchmark.c
# TODO: What is pkt_buff.c and why is gcc complaining?
# make[1]: *** No rule to make target 'pkt_buff.o', needed by 'benchmark'.  Stop.
# https://stackoverflow.com/questions/834748/gcc-makefile-error-no-rule-to-make-target
# SRCS-y := benchmark.c pkt_buff.c

CFLAGS += -g
#CFLAGS += $(WERROR_FLAGS)

include $(RTE_SDK)/mk/rte.extapp.mk
