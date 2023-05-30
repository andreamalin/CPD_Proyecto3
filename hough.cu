/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>


#include <cuda.h>
#include <string.h>
#include "CImg.h"
using namespace cimg_library;

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define M_PI 3.14159265358979323846
const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;
struct Point {
    int x;
    int y;
};

std::vector<Point> BresenhamLine(int x0, int y0, int x1, int y1) {
    std::vector<Point> points;

    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        points.push_back({x0, y0});

        if (x0 == x1 && y0 == y1) {
            break;
        }

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }

    return points;
}


//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
//TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  //TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  //TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          //TODO utilizar memoria constante para senos y cosenos
          //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados

}

//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  const char* inputFileName = argv[1];
  CImg<unsigned char> image(inputFileName);

  int *cpuht;
  int w = image.width();
  int h = image.height();

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(image.data(), w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global
  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = image.data(); // h_in contiene los pixeles de la imagen
  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int blockNum = ceil ((double)w * (double)h / (double)256);
  cudaEventRecord(start);
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
  cudaEventRecord(stop);

  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel execution time: %f ms\n", milliseconds);

  printf("Done!\n");
  // Define the threshold for line detection
  int threshold = 4000;  // Adjust this value as needed

    // Find indices of cells with accumulation values above the threshold
  std::vector<std::pair<int, int>> indices;
  for (int i = 0; i < degreeBins; i++) {
      for (int j = 0; j < rBins; j++) {
          if (h_hough[i * degreeBins + j] > threshold) {
              indices.push_back(std::make_pair(i, j));
          }
      }
  }

  // Convert indices to angle and distance values
  std::vector<float> angle_values;
  std::vector<float> distance_values;
  for (const auto& index : indices) {
      float angle = index.first * degreeInc;
      float distance = (index.second * rScale) - rMax;
      angle_values.push_back(angle);
      distance_values.push_back(distance);
  }
  
// Create a copy of the original image for drawing lines
cimg_library::CImg<unsigned char> result_image = image;


// After obtaining angle_values and distance_values
std::vector<Point> linePixels;
for (size_t i = 0; i < angle_values.size(); i++) {
    float angle = angle_values[i];
    float distance = distance_values[i];
    
    // Calculate the intersection points with the image boundaries
    int startX, startY, endX, endY;

    if (sin(angle) != 0) {
        startX = 0;
        startY = static_cast<int>(-distance / sin(angle));
        endX = result_image.width() - 1;
        endY = static_cast<int>((-distance - cos(angle) * result_image.width()) / sin(angle));
    } else {
        startX = static_cast<int>(-distance / cos(angle));
        startY = 0;
        endX = static_cast<int>((-distance - sin(angle) * result_image.height()) / cos(angle));
        endY = result_image.height() - 1;
    }

    // Calculate the real origin of the line
    int originX = startX;
    int originY = startY;

    // Adjust the starting and ending points based on the real origin
    startX -= originX;
    startY -= originY;
    endX -= originX;
    endY -= originY;

    // Bresenham's line drawing algorithm (updated with adjusted starting and ending points)
    int dx = abs(endX - startX);
    int dy = abs(endY - startY);
    int sx = (startX < endX) ? 1 : -1;
    int sy = (startY < endY) ? 1 : -1;
    int err = dx - dy;
    unsigned char gray = 200;  // Specify the color of the line

    // Calculate the length of the line
    float lineLength = sqrt(pow(endX - startX, 2) + pow(endY - startY, 2));

    while (startX != endX || startY != endY) {
        // Calculate the current pixel coordinates by adding the real origin
        int currentX = startX + originX;
        int currentY = startY + originY;

        // Check if the current pixel lies within the image boundaries
        if (currentX >= 0 && currentX < result_image.width() && currentY >= 0 && currentY < result_image.height()) {
            // Calculate the current position along the line
            float position = sqrt(pow(startX, 2) + pow(startY, 2)) / lineLength;

            // Set the color of the pixel based on the position along the line
            result_image(currentX, currentY) = gray;
        }

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            startX += sx;
        }
        if (e2 < dx) {
            err += dx;
            startY += sy;
        }
    }
    
    // Use the originX and originY variables for further processing if needed
}



// // Print the line pixels
// for (const auto& point : linePixels) {
//     std::cout << "(" << point.x << ", " << point.y << ")" << std::endl;
// }

// Display or save the resulting image with detected lines
result_image.display();  // Display the image using CImg's built-in display function
result_image.save("detected_lines.jpg");  // Save the image with detected lines to a file
  // Guardando como png
  const char* filename = "result.png";
  int stride = w;  // Apunta a escala en grises
  stbi_write_png(filename, w, h, 1, h_hough, stride);

  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_Cos);
  cudaFree(d_Sin);
  cudaFree(d_in);
  cudaFree(d_hough);
  free(h_hough);
  free(pcCos);
  free(pcSin);

  return 0;
}
