CFLAGS = -g -O3 -Wall -std=c++11

default: serial cuda 

serial: src/serial/main.cpp
	g++ ${CFLAGS} src/serial/main.cpp -o serial

cuda: src/cuda/main.cu
	nvcc -std=c++11 -arch=sm_61 src/cuda/main.cu -o cuda

clean:
	rm -rf *.o serial cuda