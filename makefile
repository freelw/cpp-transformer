DIR_INC = -I./ \
    -I./tensor \
    -I./graph \
    -I../utils/

DIR_LIB = -L./
TEST_TARGET = test
TRANSFORMER_TARGET = transformer
CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
CUDA_LIBS := -L$(CUDA_TOOLKIT)/lib64 -lcudart -lcurand
LDFLAGS = -lstdc++ $(CUDA_LIBS)
SRCDIR := ./tensor \
          ./graph \
          ./backends/cpu \
          ./backends/gpu \
          ./backends/ \
          ./optimizers \
          ./model \
		  ./module \
          ../utils/dataloader
SRCS := $(wildcard *.cpp) $(wildcard $(addsuffix /*.cpp, $(SRCDIR)))
ifeq ($(ASAN),1)
	OBJECTS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SRCS)))
else
	SRCS += $(wildcard *.cu) $(wildcard $(addsuffix /*.cu, $(SRCDIR)))
	OBJECTS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SRCS))))
endif
OBJECTS_MAIN := $(filter-out test.o transformer.o,$(OBJECTS))
OBJECTS_TEST := $(filter-out main.o transformer.o,$(OBJECTS))
OBJECTS_TRANSFORMER := $(filter-out test.o main.o,$(OBJECTS))

ifeq ($(ASAN),1)
	NVCC = g++
	NVCC_CFLAGS = -DGCC_ASAN $(DIR_INC) $(DIR_LIB) -g -fsanitize=address -fno-omit-frame-pointer
else
	NVCC = nvcc
	NVCC_CFLAGS = $(DIR_INC) $(DIR_LIB) -g -G -O3
endif

ifeq ($(RELEASE),1)
	NVCC_CFLAGS += -DNDEBUG
	NVCC_CFLAGS := $(filter-out -G,$(NVCC_CFLAGS))
endif

all: $(TEST_TARGET) $(TRANSFORMER_TARGET)

$(TRANSFORMER_TARGET) : $(OBJECTS_TRANSFORMER)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)
$(TARGET) : $(OBJECTS_MAIN)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)
$(TEST_TARGET) : $(OBJECTS_TEST)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)

%.o : %.cu
	${NVCC} -c $(NVCC_CFLAGS) $< -o $@
%.o : %.cpp
	${NVCC} -c $(NVCC_CFLAGS) $< -o $@

clean:
	@rm -f ${OBJECTS} $(TARGET) $(TEST_TARGET) $(TRANSFORMER_TARGET)

.PHONY: clean prepare_data run_test all
prepare_data:
	@cd ../../resources && \
	gunzip -c train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte && \
	gunzip -c train-images-idx3-ubyte.gz > train-images-idx3-ubyte
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