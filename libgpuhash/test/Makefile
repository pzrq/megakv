CC = gcc

CFLAGS = -Wall -g
#-D__USE_GNU -D_GNU_SOURCE 
LIBS = -lrt -lpthread -L../lib -lgpuhash

#OPENSSL_DIR = ../openssl-1.0.1c
#ifneq ($(OPENSSL_DIR), )
#	OPENSSL_LIBDIR = -L$(OPENSSL_DIR)/lib64
#	OPENSSL_INCDIR = -I$(OPENSSL_DIR)/include
#endif

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

# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L $(CUDA_LIB_PATH) -lcudart
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
ifeq ($(dbg),1)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
      TARGET    := debug
else
      TARGET    := release
endif


# Common includes and paths for CUDA
NVCCINCLUDES      := -I$(CUDA_INC_PATH) -I. -I/usr/local/cuda/samples/common/inc/

################################################################

HEADER_DIR = ../
OBJECT_DIR = objs
TARGET_DIR = .

TARGET_FILE = run
TARGET = $(addprefix $(TARGET_DIR)/, $(TARGET_FILE))

SOURCE_FILES = $(wildcard *.c)
HEADER_FILES = $(wildcard *.h) $(wildcard *.h)
OBJECT_FILE = $(SOURCE_FILES:.c=.o)

OBJECT = $(addprefix $(OBJECT_DIR)/, $(OBJECT_FILE))

all: $(TARGET)

$(TARGET): $(OBJECT_DIR) $(OBJECT) | $(TARGET_DIR)
	$(CC) -g  $(OBJECT) -o $@ $(LIBS) $(LDFLAGS)

$(OBJECT_DIR):
	mkdir $(OBJECT_DIR)


$(OBJECT_DIR)/%.o : %.c
	$(CC) $(CFLAGS) -I $(HEADER_DIR) -I $(CUDA_INC_PATH) -c $< -o $@

.PHONY : clean
clean:
	rm -rf $(TARGET) $(OBJECT)
