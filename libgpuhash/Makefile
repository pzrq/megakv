####################################################
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
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             := gcc

# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH) 
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

# Debug build flags
dbg = 1
ifeq ($(dbg),1)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
      TARGET    := debug
else
      TARGET    := release
endif

NVCCFLAGS += -Xptxas -dlcm=cg


# Common includes and paths for CUDA
NVCCINCLUDES      := -I$(CUDA_INC_PATH) -I. -I/usr/local/cuda/samples/common/inc/

####################################################


OBJS_DIR = objs
TARGET_DIR = lib
TARGET_FILE = libgpuhash.a
TARGET = $(addprefix $(TARGET_DIR)/, $(TARGET_FILE))

.SUFFIXES : .cu .c .o

CU_SRC_FILES = $(wildcard *.cu)
CC_SRC_FILES = $(wildcard *.c)
HEADER_FILES = $(wildcard *.h) $(wildcard *.h)

SRC_FILES = $(CU_SRC_FILES) $(CC_SRC_FILES)
OBJS_FILE = $(CU_SRC_FILES:.cu=.o)

OBJS = $(addprefix $(OBJS_DIR)/, $(OBJS_FILE))
DEPS = Makefile.dep

all: $(TARGET) 

$(TARGET): $(DEPS) $(OBJS_DIR) $(OBJS) | $(TARGET_DIR)
	ar rcs $@ $(OBJS)

$(TARGET_DIR):
	mkdir $(TARGET_DIR)

$(OBJS_DIR):
	mkdir $(OBJS_DIR)

$(DEPS): $(SRC_FILES) $(HEADER_FILES)
	$(CC) -MM -MP -x c++ $(CU_SRC_FILES) $(CC_SRC_FILES) | sed 's![^:]*.o:!objs/&!g' > Makefile.dep

#$(OBJS_DIR)/%.o : %.c
#	$(GCC) $(CCFLAGS) $(NVCCINCLUDES) -c $< -o $@

$(OBJS_DIR)/%.o : %.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -c $< -o $@

.PHONY : clean


clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)

ifneq ($(MAKECMDGOALS), clean)
-include $(DEPS)
endif
