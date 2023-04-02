CXX = g++-12
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp
CBUGFLAGS = --enable-checking -std=c++11 -g -O0 -march=native -fopenmp


TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

p2 :
	g++-12 -std=c++11 -g -O0 -march=native -fopenmp omp-scan.cpp -o omp-scan

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
