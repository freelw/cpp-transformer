DIR_INC = -I./ \
    -I./tensor \
    -I./graph

#DIR_LIB = -L./
DIR_LIB = 
TEST_TARGET = test
TRANSFORMER_TARGET = transformer
LM_TARGET = lm
MNIST_TARGET = handwritten_recognition
CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
CUDA_LIBS := -L$(CUDA_TOOLKIT)/lib64 -lcudart -lcurand
LDFLAGS =
ASAN_FLAGS = -fsanitize=address
SRCDIR := ./tensor \
          ./graph \
          ./backends/cpu \
          ./backends/gpu/cuda \
		  ./backends/gpu/metal \
          ./backends/ \
          ./optimizers \
          ./model \
          ./module \
          ./module/translation \
          ./module/language_model \
          ./dataloaders/translation \
          ./dataloaders/language_model \
		  ./dataloaders/mnist \
          ./dataloaders \
          ../utils/dataloader
SRCS := $(wildcard *.cpp) $(wildcard $(addsuffix /*.cpp, $(SRCDIR)))
CPU ?= $(ASAN)
ifneq ($(CPU),1)
ifneq ($(MACOS),1)
CUDA_GPU := 1
endif
endif

OBJECTS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SRCS)))
ifeq ($(CUDA_GPU), 1)
	SRCS += $(wildcard *.cu) $(wildcard $(addsuffix /*.cu, $(SRCDIR)))
	OBJECTS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SRCS))))
	LDFLAGS += $(CUDA_LIBS)
endif

OBJECTS_TEST := $(filter-out transformer.o lm.o mnist.o,$(OBJECTS))
OBJECTS_TRANSFORMER := $(filter-out test.o lm.o mnist.o,$(OBJECTS))
OBJECTS_LM := $(filter-out test.o transformer.o mnist.o,$(OBJECTS))
OBJECTS_MNIST := $(filter-out test.o transformer.o lm.o,$(OBJECTS))

ifeq ($(CPU),1)
	NVCC = g++
	NVCC_CFLAGS = -DGCC_CPU $(DIR_INC) $(DIR_LIB) -g -fno-omit-frame-pointer
else
	NVCC = nvcc
	NVCC_CFLAGS = $(DIR_INC) $(DIR_LIB) -g -O3
	ifeq ($(MACOS),1)
		NVCC = g++
		DIR_INC += -I./backends/gpu/metal/metal-cpp
		FRAMEWORKS = -framework Metal -framework Foundation -framework QuartzCore
		NVCC_CFLAGS += -DMETAL_GPU -std=c++17
		LDFLAGS += $(FRAMEWORKS)
	else
		NVCC_CFLAGS += -DCUDA_GPU -G -std=c++17
	endif
endif

ifeq ($(ASAN),1)
	NVCC_CFLAGS += $(ASAN_FLAGS)
endif

ifeq ($(RELEASE),1)
	NVCC_CFLAGS += -DNDEBUG
	NVCC_CFLAGS := $(filter-out -G,$(NVCC_CFLAGS))
else
	NVCC_CFLAGS := $(filter-out -O3,$(NVCC_CFLAGS))
endif

ifeq ($(MACOS), 1)
	SDK_PATH := $(shell xcrun --sdk macosx --show-sdk-path)
	NVCC_CFLAGS += -isysroot $(SDK_PATH)
endif

all: $(TEST_TARGET) $(TRANSFORMER_TARGET) $(LM_TARGET) $(MNIST_TARGET)

$(TRANSFORMER_TARGET) : $(OBJECTS_TRANSFORMER)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)
$(LM_TARGET) : $(OBJECTS_LM)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)
$(TEST_TARGET) : $(OBJECTS_TEST)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)
$(MNIST_TARGET) : $(OBJECTS_MNIST)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)

%.o : %.cu
	${NVCC} -c $(NVCC_CFLAGS) $< -o $@
%.o : %.cpp
	${NVCC} -c $(NVCC_CFLAGS) $< -o $@

clean:
	@rm -f ${OBJECTS} ${TEST_TARGET} ${TRANSFORMER_TARGET} ${MNIST_TARGET}

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