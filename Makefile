CC=mpic++
OUTPUT=fletcher.x

compile:
	$(CC) main.cpp -o $(OUTPUT)
clean:
	rm *.x