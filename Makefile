CFLAGS = -pthread -O3 -Wall -std=c++11

default: serial #parallel 

serial:
	g++ ${CFLAGS} src/serial/main.cpp -o serial

#parallel:
#	g++ ${CFLAGS} src/serial/main.cpp -o parallel_softmax

clean:
	rm -rf *.o serial_softmax parallel_softmax 