# Linux Makefile
#
# @author Simion-Constantinescu Andrei
#

.PHONY: all, clean

CXX      = g++
CPPFLAGS = -Wall -std=c++11
OBJ_LIST = test1.o
EXEC	 = test1
RM       = rm -f

all: build

build: $(EXEC)

$(EXEC): $(OBJ_LIST)
	$(CXX) $^ -o $@

test1.o: src/test1.cpp
	$(CC) $(CPPFLAGS) -c $<

clean:
	$(RM) *.o $(EXEC)