DIR_INC = -I./ \
    -I./tensor \
    -I./graph \
    -I../utils/

DIR_LIB = -L./
TEST_TARGET = test
TRANSFORMER_TARGET = transformer
LM_TARGET = lm
CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
CUDA_LIBS := -L$(CUDA_TOOLKIT)/lib64 -lcudart -lcurand
LDFLAGS = -lstdc++
ASAN_FLAGS = -fsanitize=address
SRCDIR := ./tensor \
          ./graph \
          ./backends/cpu \
          ./backends/gpu \
          ./backends/ \
          ./optimizers \
          ./model \
		  ./module \
		  ./module/translation \
          ../utils/dataloader
SRCS := $(wildcard *.cpp) $(wildcard $(addsuffix /*.cpp, $(SRCDIR)))
CPU ?= $(ASAN)
ifeq ($(CPU),1)
	OBJECTS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SRCS)))
else
	SRCS += $(wildcard *.cu) $(wildcard $(addsuffix /*.cu, $(SRCDIR)))
	OBJECTS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SRCS))))
	LDFLAGS += $(CUDA_LIBS)
endif

OBJECTS_TEST := $(filter-out transformer.o lm.o,$(OBJECTS))
OBJECTS_TRANSFORMER := $(filter-out test.o lm.o,$(OBJECTS))
OBJECTS_LM := $(filter-out test.o transformer.o,$(OBJECTS))

ifeq ($(CPU),1)
	NVCC = g++
	NVCC_CFLAGS = -DGCC_CPU $(DIR_INC) $(DIR_LIB) -g -fno-omit-frame-pointer
else
	NVCC = nvcc
	NVCC_CFLAGS = $(DIR_INC) $(DIR_LIB) -g -G -O3
endif

ifeq ($(ASAN),1)
	NVCC_CFLAGS += $(ASAN_FLAGS)
endif

ifeq ($(RELEASE),1)
	NVCC_CFLAGS += -DNDEBUG
	NVCC_CFLAGS := $(filter-out -G,$(NVCC_CFLAGS))
endif

ifeq ($(MACOS), 1)
	SDK_PATH := $(shell xcrun --sdk macosx --show-sdk-path)
	NVCC_CFLAGS += -isysroot $(SDK_PATH)
endif

all: $(TEST_TARGET) $(TRANSFORMER_TARGET) $(LM_TARGET)

$(TRANSFORMER_TARGET) : $(OBJECTS_TRANSFORMER)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)
$(LM_TARGET) : $(OBJECTS_LM)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)
$(TEST_TARGET) : $(OBJECTS_TEST)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)

%.o : %.cu
	${NVCC} -c $(NVCC_CFLAGS) $< -o $@
%.o : %.cpp
	${NVCC} -c $(NVCC_CFLAGS) $< -o $@

clean:
	@rm -f ${OBJECTS} ${TEST_TARGET} ${TRANSFORMER_TARGET}

.PHONY: clean run_test all

dbg:
	@echo "CFLAGS is $(CFLAGS)"
	@echo "SRCS is $(SRCS)"
	@echo "OBJECTS is $(OBJECTS)"
	@echo "SRCDIR is $(SRCDIR)"
	@echo "DIR_INC is $(DIR_INC)"
	@echo "DIR_LIB is $(DIR_LIB)"
	@echo "TARGET is $(TARGET)"
	@echo "NVCC is $(NVCC)"
run_test: $(TEST_TARGET)
	@echo "Test cpu"
	@./$(TEST_TARGET) -t 0
	@echo "Test cpu done"
	@echo "Test gpu"
	@./$(TEST_TARGET) -t 1
	@echo "Test gpu done"
	@echo "Test completed"