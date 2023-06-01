all: hough

hough:	hough.cu
	nvcc hough.cu -o hough -I/msys64/home/andre/CImg-3.2.5_pre051823  user32.lib gdi32.lib
	./hough cuadrosHough.bmp result.bmp

