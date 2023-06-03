all: hough_shared hough_constant hough

hough_shared:	hough_shared.cu
	nvcc hough_shared.cu -o hough_shared -I/Cimg.h user32.lib gdi32.lib
	./hough_shared cuadrosHough.bmp result.bmp

hough_constant:	hough_constant.cu
	nvcc hough_constant.cu -o hough_constant -I/Cimg.h user32.lib gdi32.lib
	./hough_constant cuadrosHough.bmp result.bmp

hough:	hough.cu
	nvcc hough.cu -o hough -I/Cimg.h user32.lib gdi32.lib
	./hough cuadrosHough.bmp result.bmp

