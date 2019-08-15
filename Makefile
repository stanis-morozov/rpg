CC=g++
CFLAGS=-fopenmp -O3 -march=native --std=c++17
DEPS=hnswalg.h  hnswlib.h  L2space.h  visited_list_pool.h

all: RPG prepare

RPG: RPG.cpp $(DEPS)
	$(CC) RPG.cpp -o RPG $(CFLAGS)

prepare: prepare.cpp
	$(CC) prepare.cpp -o prepare $(CFLAGS)

clean:
	rm prepare RPG
