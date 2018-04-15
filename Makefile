CFLAGS = -pthread -g -O3 -Wall -std=c++11

default: serial #parallel 

serial: src/serial/main.cpp
	g++ ${CFLAGS} src/serial/main.cpp -o serial

#parallel:
#	g++ ${CFLAGS} src/serial/main.cpp -o parallel

clean:
	rm -rf *.o serial parallel