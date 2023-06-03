all: hough_shared

hough_shared:	hough_shared.cu
	nvcc hough_shared.cu -o hough_shared -I/msys64/home/Brand/CImg-3.2.5_pre051823 user32.lib gdi32.lib
	./hough_shared cuadrosHough.bmp result.bmp

