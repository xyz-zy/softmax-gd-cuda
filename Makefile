CFLAGS = -g -O3 -Wall -std=c++11
CUDAFLAGS = -g -O3 -std=c++11 -arch=sm_61 -rdc=true -lcudadevrt

default: serial cuda 

serial: src/serial/main.cpp
	g++ ${CFLAGS} src/serial/main.cpp -o serial

cuda: src/cuda/main.cu
	nvcc ${CUDAFLAGS} src/cuda/main.cu -o cuda

clean:
	rm -rf *.o serial cuda
