NVCC = nvcc
TARGET_EXEC ?= a.out

#FREESTAND_DIR ?= ./freestanding
BUILD_DIR ?= ./build
SRC_DIRS ?= ./src
INC_DIRS ?= ./include
EXE_DIR ?= $(BUILD_DIR)/exec

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s -or -name *.cu)
SRCS_NAMES := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s -or -name *.cu -printf "%f\n")
HEADER_SRCS_NAMES := $(shell find ./include/src -name *.cpp -printf "%f\n")
OBJS := $(SRCS:%=$(BUILD_DIR)/obj/%.o)
EXES := $(SRCS:%=$(BUILD_DIR)/exe/%.exe)
DEPS := $(OBJS:.o=.d)
HEADER_OBJS := $(HEADER_SRCS_NAMES:%=$(BUILD_DIR)/include/obj/%.o)
HEADER_DEPS := $(HEADER_OBJS:.o=.d)

#INCL_DIRS := $(shell find $(INC_DIRS) -type d) ./include $(FREESTAND_DIR)/include 
INCL_DIRS := ./include ../3rdparty/fftw-cufftw-benchmark/Common/
#
#$(FREESTAND_DIR)/include 
INC_FLAGS := $(addprefix -I,$(INCL_DIRS))
LDFLAGS :=  -lcurand -lcufft
CPPFLAGS ?= $(INC_FLAGS) -Wall -pthread -MMD -MP -shared -fPIC -std=c++11 -O3 -mavx -ftree-vectorize -fopt-info-vec
CUDAFLAGS = $(INC_FLAGS) -res-usage -lineinfo -maxrregcount 40  --restrict --extended-lambda -std=c++17 -O3 -DCUDA -DNOT_IMPL -arch=sm_86 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86
# -opt-info inline -res-usage
#  -maxrregcount 128
#  -Xptxas --def-load-cache=cs -Xptxas --def-store-cache=cs

all: objs exes header_objs

objs: $(OBJS)

header_objs: $(HEADER_OBJS)

exes: $(EXES)

$(BUILD_DIR)/exe/%.exe: $(BUILD_DIR)/obj/%.o  $(HEADER_OBJS)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $< $(HEADER_OBJS) -o $@ $(LDFLAGS)

# assembly
$(BUILD_DIR)/obj/%.s.o: %.s
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) -c $< -o $@

# c source
$(BUILD_DIR)/obj/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# c++ source
$(BUILD_DIR)/obj/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# c++ header source
#$(BUILD_DIR)/include/obj/NFALoader.cpp.o: include/src/NFALoader.cpp
#	$(MKDIR_P) $(dir $@)
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/include/obj/%.cpp.o: include/src/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# cuda source
$(BUILD_DIR)/obj/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)
-include $(HEADER_DEPS)

MKDIR_P ?= mkdir -p