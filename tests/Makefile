all: pgm.o	hough

hough:	hough.cu pgm.o 
	nvcc hough.cu pgm.cpp -o hough -I/msys64/home/andre/CImg-3.2.5_pre051823  user32.lib gdi32.lib
	./hough runway.pgm result.png

pgm.o:	pgm.cpp
	g++ -c pgm.cpp -o ./pgm.o
