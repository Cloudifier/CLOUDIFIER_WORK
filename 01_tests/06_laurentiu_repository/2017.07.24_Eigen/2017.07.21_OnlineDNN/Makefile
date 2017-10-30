# Linux Makefile
#
# @author Simion-Constantinescu Andrei
#

.PHONY: all, clean

CXX      = g++
CPPFLAGS = -Wall -std=c++11 -Iinclude
OBJ_LIST = main.o
EXEC	 = main
RM       = rm -f

all: build

build: $(EXEC)

$(EXEC): $(OBJ_LIST)
	$(CXX) $^ -o $@

main.o: main.cpp online_dnn_engine.hpp
	$(CC) $(CPPFLAGS) -c $<

clean:
	$(RM) *.o $(EXEC)