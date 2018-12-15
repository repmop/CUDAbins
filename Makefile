CU_EXECUTABLE := cudabins

EXECUTABLE := bins

CU_FILES   := cudabins.cu

CU_DEPS    :=

CC_FILES   := main.cpp bin.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs

CXX=g++ -m64 -std=c++11
# CXX=g++ -m64 -std=c++11

CXXFLAGS=-O3 -Wall
ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib/ -lcudart
else
# Building on Linux
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -L/usr/local/cuda/lib64/ -lcudart
endif


NVCC=nvcc
# NVCC=nvcc --compiler-options -Wall

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_60 -std=c++11
OBJS=$(OBJDIR)/main.o  $(OBJDIR)/bin.o $(OBJDIR)/parse.o
OBJS_CU=$(OBJDIR)/main.o  $(OBJDIR)/cudabins.o $(OBJDIR)/parse.o


.PHONY: dirs clean

default: $(EXECUTABLE) $(CU_EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(CU_EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(CU_EXECUTABLE): dirs $(OBJS_CU)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CU) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
