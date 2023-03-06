CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native

RM = rm -f
MKDIRS = mkdir -p

TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

compute-vectorize:
	g++ -std=c++11 -O3 -march=native compute.cpp -ftree-vectorize -fopt-info-vec-optimized -o compute

%:%.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	$(RM) $(TARGETS)

