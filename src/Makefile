ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif

ifeq ($(FIRST_MARK),)
CUR_DIR = $(shell pwd)
LIBGPUHASHDIR = $(CUR_DIR)/../libgpuhash/
FIRST_MARK = 1
export FIRST_MARK
export CUR_DIR
export LIBGPUHASHDIR
endif

# Default target, can be overriden by command line or environment
RTE_TARGET ?= x86_64-native-linuxapp-gcc

include $(RTE_SDK)/mk/rte.vars.mk

# binary name
APP = megakv

# all source are stored in SRCS-y
SRCS-y := mega.c items.c mega_common.c  mega_memory.c  mega_recv.c  mega_send.c  mega_timer.c slabs.c bitmap.c mega_scheduler.c dpdk.c


LDFLAGS += -lrt -lpthread -L$(LIBGPUHASHDIR)lib -lgpuhash

CFLAGS += -g -Wall
#CFLAGS += $(WERROR_FLAGS)

# Common includes and paths for CUDA
################################################################
# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
#CUDA_SDK_DIR = /home/$(USER)/NVIDIA_CUDA-5.5_Samples/

ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

ifneq ($(DARWIN),)
      LDFLAGS   += -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   += -L$(CUDA_LIB_PATH) -lcudart
  else
      LDFLAGS   += -L$(CUDA_LIB_PATH) -lcudart
  endif
endif
# Common includes and paths for CUDA
CFLAGS += -I$(CUDA_INC_PATH) #-I$(CUDA_SDK_DIR)common/inc/


################################################################


CFLAGS += -I$(CUR_DIR)/include/
CFLAGS += -I$(LIBGPUHASHDIR)

CFLAGS += -D_GNU_SOURCE -DMEGA_MALLOC
# AFFINITY_1: no-hyperthreading, on CPU 1
# AFFINITY_2: hyperthreading, on CPU 1
# TODO: Disabled CPU affinity so this runs on AWS VMs, is that reasonable?
# CFLAGS += -DCPU_AFFINITY -DAFFINITY_4
CFLAGS += -DNOT_FORWARD_0 -DNOT_COLLECT_0 -DNOT_GPU_0
CFLAGS += -DRECEIVER_PERFORMANCE_TEST_0
CFLAGS += -DCOMPACT_JOB -DKEY_MATCH_0
CFLAGS += -DUSE_LOCK
CFLAGS += -DTWO_PORTS_0
CFLAGS += -DSIGNATURE_0
CFLAGS += -DPRELOAD
CFLAGS += -DPREFETCH_PIPELINE_0
CFLAGS += -DPREFETCH_BATCH

CFLAGS += -DLOCAL_TEST
# defined only under LOCAL_TEST
#CFLAGS += -DDIS_ZIPF
CFLAGS += -DDIS_UNIFORM


include $(RTE_SDK)/mk/rte.extapp.mk

