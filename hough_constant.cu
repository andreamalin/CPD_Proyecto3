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

#define M_PI 3.14159265358979323846
#define DEG2RAD (M_PI/180.0f)
const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;
//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[(int)((round(rMax * 2 *180)))];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * (int)((round(rMax * 2 *180)))); //init en ceros

  int xCent = w / 2;
  int yCent = h / 2;

  for (int x = 0; x < w; x++) //por cada pixel
    for (int y = 0; y < h; y++) //...
      {
        int idx = (y * w) + x;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            float xCoord = x - xCent;
            float yCoord = y - yCent; 
            //printf("CPU: ID: %i; X: %f, Y: %f", idx, xCoord, yCoord);
            for (int theta = 0; theta < rMax; theta++)
              {
                float distance = ( (xCoord) * cos((float)theta * DEG2RAD)) + ((yCoord) * sin((double)theta * DEG2RAD));
                (*acc)[ (int)((round(distance + rMax) * 180)) + theta]++; //+1 para este radio distance y este theta
              }
          }
      }
}

//*****************************************************************
// DONE usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

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
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  //TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  int xCoord = gloID % w - xCent;
  int yCoord = (yCent - gloID / w) * -1;

  //TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0)
    {
      // printf("GPU: ID: %i; X: %i, Y: %i", gloID, xCoord, yCoord);
      for (int theta = 0; theta < rMax; theta++)
        {
          //DONE utilizar memoria constante para senos y cosenos
          float distance = (xCoord * cosf((float)theta * DEG2RAD)) + (yCoord * sinf((double)theta * DEG2RAD)); //probar con esto para ver diferencia en tiempo
          // float distance = (xCoord * d_Cos[theta] * DEG2RAD) + (yCoord * d_Sin[theta] * DEG2RAD);
          //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
          atomicAdd(&acc[(int)((round(distance + rMax) * 180)) + theta], 1); //+1 para este radio distance y este theta
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
  const char* outputFileName = argv[2];

  CImg<unsigned char> image(inputFileName);

  int *cpuht;
  int w = image.width();
  int h = image.height();

  // float* d_Cos;
  // float* d_Sin;

  // cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  // cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

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
  cudaMemcpyToSymbol(d_Cos, pcCos, sizeof (float) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, pcSin, sizeof (float) * degreeBins);

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global
  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = image.data(); // h_in contiene los pixeles de la imagen
  h_hough = (int *) malloc (sizeof (int) * (int)((round(rMax * 2 * 180))));

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
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
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


  // Guardando como png
  CImg<unsigned char> result_image = image;
  unsigned char red[] = {255, 0, 0};

  int threshold = 200;  // AJUSTAR SEGUN LO NECESARIO
  // Guardando los valores arriba del threshold

  float _accu_h = ((sqrt(2.0) * (double)(h>w?h:w)) / 2.0) * 2; 
  std::vector<std::pair<int, int>> indices;
  for(int r=0;r<_accu_h;r++) {
    for(int t=0;t<180;t++) {
      if((int)cpuht[(r*180) + t] >= threshold) {
        indices.push_back(std::make_pair(r, t));
      }
    }
  }

    
  for (const auto& index : indices) {
    int r = index.first;
    int t = index.second;
    int x1, y1, x2, y2;
    x1 = y1 = x2 = y2 = 0;

    if(t >= 45 && t <= 135)
    {
      //y = (r - x cos(t)) / sin(t)
      x1 = 0;
      y1 = ((double)(r-(_accu_h/2)) - ((x1 - (w/2) ) * cos(t * DEG2RAD))) / sin(t * DEG2RAD) + (h / 2);
      x2 = w - 0;
      y2 = ((double)(r-(_accu_h/2)) - ((x2 - (w/2) ) * cos(t * DEG2RAD))) / sin(t * DEG2RAD) + (h / 2);
    }
    else
    {
      //x = (r - y sin(t)) / cos(t);
      y1 = 0;
      x1 = ((double)(r-(_accu_h/2)) - ((y1 - (h/2) ) * sin(t * DEG2RAD))) / cos(t * DEG2RAD) + (w / 2);
      y2 = h - 0;
      x2 = ((double)(r-(_accu_h/2)) - ((y2 - (h/2) ) * sin(t * DEG2RAD))) / cos(t * DEG2RAD) + (w / 2);
    }

    result_image.draw_line(x1, y1, x2, y2, red);
  }

  // Display or save the resulting image with detected lines
  result_image.display();  // Display the image using CImg's built-in display function
  result_image.save(outputFileName);  // Save the image with detected lines to a file

  
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
