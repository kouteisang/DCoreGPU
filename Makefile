# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -O3
INCLUDES = -I./Graph

# Target executable
TARGET = main

# Source files
CUDA_SRCS = main.cu src/gpubaseline.cu src/klist.cu src/klistprune.cu src/klistanchorbinary.cu src/klistanchorbinaryprune.cu src/klistanchorsequenceprune.cu src/klist_balance.cu src/klist_balance_buffer.cu
CPP_SRCS = Graph/Graph.cpp

# Rules
all:
	$(NVCC) $(CUDA_SRCS) $(CPP_SRCS) -o $(TARGET) $(INCLUDES)

clean:
	rm -f $(TARGET)
