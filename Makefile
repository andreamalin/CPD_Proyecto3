all: pgm.o	hough

hough:	hough.cu pgm.o
	nvcc hough.cu pgm.cpp -o hough
	./hough cuadrosHough.pgm

pgm.o:	pgm.cpp
	g++ -c pgm.cpp -o ./pgm.o
