CXX := g++
CXXFLAGS := -lpthread -O3

all: pi.out pi_ori.out

pi.out : pi.cpp
	$(CXX) $^ $(CXXFLAGS)  -o $@

pi_ori.out : pi_ori.cpp
	$(CXX) $^ $(CXXFLAGS)  -o $@

clean:
	rm  pi.out pi_ori.out
