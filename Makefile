CC=nvcc
WRAPPER=mpic++
OUTPUT=fletcher.x
INCLUDE=-I/usr/local/cuda/include
LIB=-L/usr/local/cuda/lib64
FLAGS=-lcudart -lcuda
compile:
	$(CC) -ccbin $(WRAPPER) main.cpp -o $(OUTPUT) $(INCLUDE) $(LIB) $(FLAGS)
clean:
	rm *.x
