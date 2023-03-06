CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native

RM = rm -f
MKDIRS = mkdir -p

TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

solve_test01:
	g++ -g val_test01_solved.cpp -o val_test01_solved && valgrind --leak-check=full --track-origins=yes ./val_test01_solved

solve_test02:
	g++ -g val_test02_solved.cpp -o val_test02_solved && valgrind --leak-check=full --track-origins=yes ./val_test02_solved

%:%.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	$(RM) $(TARGETS)

