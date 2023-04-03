CXX = g++-12
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp
CBUGFLAGS = --enable-checking -std=c++11 -g -O0 -march=native -fopenmp

TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

p2 :
	g++-12 -std=c++11 -g -O0 -march=native -fopenmp omp-scan.cpp -o omp-scan

ja : jacobi2D-omp.cpp
	g++-12 -std=c++11 -g -O3 -march=native -fopenmp jacobi2D-omp.cpp -o jacobi2D-omp

gs : gs2D-omp.cpp
	g++-12 -std=c++11 -g -O3 -march=native -fopenmp  gs2D-omp.cpp -o gs2D-omp

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
