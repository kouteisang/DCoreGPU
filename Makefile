# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -O2
INCLUDES = -I./Graph

# Target executable
TARGET = main

# Source files
CUDA_SRCS = main.cu
CPP_SRCS = Graph/Graph.cpp

# Rules
all:
	$(NVCC) $(CUDA_SRCS) $(CPP_SRCS) -o $(TARGET) $(INCLUDES)

clean:
	rm -f $(TARGET)
