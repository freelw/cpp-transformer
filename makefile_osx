CC = gcc
CXX = g++

DIR_INC = -I./
DIR_INC += -I./layers
DIR_INC += -I./dataloader
DIR_INC += -I./checkpoint
DIR_INC += -I./test

DIR_LIB = -L./

TARGET = transformer

# macOS system SDK and libomp paths
SDK_PATH := $(shell xcrun --sdk macosx --show-sdk-path)
OMP_INC = -I/opt/homebrew/opt/libomp/include
OMP_LIB = -L/opt/homebrew/opt/libomp/lib

# Clean CFLAGS: no linker flags here
CFLAGS = -std=c++11 -g -Wall $(DIR_INC) \
  -fsanitize=address \
  -Xpreprocessor -fopenmp $(OMP_INC) \
  -fno-omit-frame-pointer \
  -isysroot $(SDK_PATH)

# Clean LDFLAGS: includes libomp + project-specific lib path
LDFLAGS = $(OMP_LIB) $(DIR_LIB) -lomp

SRCDIR += ./matrix
SRCDIR += ./autograd
SRCDIR += ./stats
SRCDIR += ./layers
SRCDIR += ./dataloader
SRCDIR += ./checkpoint
SRCDIR += ./test

SRCS := $(wildcard *.cpp) $(wildcard $(addsuffix /*.cpp, $(SRCDIR)))
OBJECTS := $(SRCS:.cpp=.o)

RELEASE_MSG = "[warning!!!!!] Compiling with debug flags"
ifeq ($(RELEASE),1)
	CFLAGS += -O3
	CFLAGS := $(filter-out -fsanitize=address, $(CFLAGS))
	RELEASE_MSG = "Compiling with optimizations for release"
else
	CFLAGS += -fsanitize=address
endif

$(TARGET): $(OBJECTS)
	@echo $(RELEASE_MSG)
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

clean:
	@rm -f $(OBJECTS) $(TARGET)

.PHONY: clean