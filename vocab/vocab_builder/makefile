DIR_INC = -I./
DIR_LIB = -L./
TARGET	= builder
CFLAGS = -g -Wall $(DIR_INC) $(DIR_LIB) -fsanitize=address -fopenmp
LDFLAGS += -lstdc++ -fopenmp
SRCS := $(wildcard *.cpp) $(wildcard $(addsuffix /*.cpp, $(SRCDIR)))
OBJECTS := $(patsubst %.c,%.o,$(SRCS))
RELEASE_MSG="[warning!!!!!] Compiling with debug flags"
ifeq ($(RELEASE),1)
	CFLAGS += -O3
	CFLAGS := $(filter-out -fsanitize=address, $(CFLAGS))
	RELEASE_MSG = "Compiling with optimizations for release"
else
	CFLAGS += -fsanitize=address
endif

$(TARGET) : $(OBJECTS)
	@echo $(RELEASE_MSG)
	g++ $(CFLAGS) $^ -o $@ $(LDFLAGS)
%.o : %.c
	g++ -c $(CFLAGS) $< -o $@
clean:
	@rm -f *.o $(TARGET)
.PHONY:clean