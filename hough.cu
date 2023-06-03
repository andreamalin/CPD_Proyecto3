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

/*
  M_PI hace referencia al valor numérico Pi
  DEG2RAG es la constante para convertir grados a radianes
  degreeBins representa la cantidad de bins en los que se dividen
  180 grados en el acumulador utilizado en la transformada de Hough
*/
#define M_PI 3.14159265358979323846
#define DEG2RAD (M_PI/180.0f)
const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;

//*****************************************************************
/*
  Transformada de Hough en el CPu.
  Se identifican los pixeles casi blancos y se actualiza el
  acumulador en función de las líneas detectadas.
*/
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  /*
    La distancia máxima desde el centro de la imagen hasta
    una esquina, haciendo uso de pitágoras
  */
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  /*
    El tamaño del acumulador se determina multiplicando el radio
    máximo por 2 y por 180 (grados) (2 grados por bin)
  */
  *acc = new int[(int)((round(rMax * 2 *180)))]; 
  memset (*acc, 0, sizeof (int) * (int)((round(rMax * 2 *180))));

  /**
   * Centro de la imagen horizontal y verticalmente
  */
  int xCent = w / 2;
  int yCent = h / 2;

  /**
   * Se recorre cada píxel de la imagen. Si el valor del
   * píxel es mayor a 100 (considerado alejado de negro -> blanco),
   * se procede a realizar la transformada de Hough para ese píxel.
  */
  for (int x = 0; x < w; x++)
    for (int y = 0; y < h; y++) 
      {
        int idx = (y * w) + x;
        
        if (pic[idx] > 100) 
          {
            float xCoord = x - xCent;
            float yCoord = y - yCent; 
            /*
              Se calculan las coordenadas relativas al centro de la imagen y
              se calcula r = x.cos(theta) + y.sin(theta). Se incrementa el acumulador.
            */
            for (int theta = 0; theta < rMax; theta++) {
              float distance = ( (xCoord) * cos((float)theta * DEG2RAD)) + ((yCoord) * sin((double)theta * DEG2RAD));  
              (*acc)[ (int)((round(distance + rMax) * 180)) + theta]++; 
            }
          }
      }
}

/*
  Transformada de Hough en la GPU. Se realiza en paralelo
  haciendo uso de multiples hilos, donde cada hilo procesa un pixel.
*/
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  /*
    Se calcula el ID basado en el indice de bloque e hilo.
    Si es mayor que el tamaño de la imagen, no se realiza nada.
  */
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;
  
  /**
   * Centro de la imagen horizontal y verticalmente
  */
  int xCent = w / 2;
  int yCent = h / 2;

  /*
    Se calculan las coordenadas relativas al centro de la imagen
  */
  int xCoord = gloID % w - xCent;
  int yCoord = (yCent - gloID / w) * -1;

  /**
   * Si el valor del píxel es mayor a 250 (considerado casi blanco),
   * se procede a realizar la transformada de Hough para ese píxel.
  */
  if (pic[gloID] > 100)
  {
    /*
      Se calcula r = x.cos(theta) + y.sin(theta). Se incrementa el acumulador.
    */
    for (int theta = 0; theta < rMax; theta++)
    {
      float distance = ( (xCoord) * cos((float)theta * DEG2RAD)) + ((yCoord) * sin((double)theta * DEG2RAD));  
      /*
        Se hace uso de atomicAdd para manejar actualizaciones concurrentes de múltiples hilos.
        Evitamos race conditions, al hacer que nuestra suma se realice de forma atómica,
        es decir, que no sea afectada por otros hilos que puedan estar realizando la misma
        operación en la misma ubicación de memoria. Así, no se mezclan resultados.
      */
      atomicAdd(&acc[(int)((round(distance + rMax) * 180)) + theta], 1); 
    }
  }
}

//*****************************************************************
int main (int argc, char **argv)
{
  int i;
  /*
    En los argumentos, se obtiene la imagen a procesar y el nombre
    de la imagen final.
  */
  const char* inputFileName = argv[1];
  const char* outputFileName = argv[2];
  /*
    Se abre la imagen haciendo uso de CImg
    Se obtiene su ancho u altura
  */
  CImg<unsigned char> image(inputFileName);
  int *cpuht;
  int w = image.width();
  int h = image.height();
  
  // CPU calculation
  CPU_HoughTran(image.data(), w, h, &cpuht);

  /*
    Se reserva memoria en la GPU para los arreglos
    d_in (datos de entrada de la imagen) y d_hough (acumulador en la GPU).
  */
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / 180;

  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = image.data(); // h_in contiene los pixeles de la imagen
  h_hough = (int *) malloc (sizeof (int) * (int)((round(rMax * 2 * 180))));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * 180);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * 180);

  /**
   * Se toman los tiempos de ejecucion haciendo uso de un cudaEvent
   */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  /*
    Se configura la cantidad de bloques de acuerdo al tamaño de la imagen
  */
  int blockNum = ceil ((double)w * (double)h / (double)256);
  cudaEventRecord(start);
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);

  cudaEventRecord(stop);

  /*
    Se obtienen los resultados y, se comparan contra los del CPU
  */
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * 180, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  for (i = 0; i < degreeBins * 180; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }

  /*
    Se muestra el tiempo de ejecucion en pantalla
  */
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel execution time: %f ms\n", milliseconds);
  printf("Done!\n");

  /**
   * Se muestra el resultado en pantalla. Para cada índice en el vector,
   * se revisa si es mayor que el threshold. Si lo es, se guarda la (distancia, angulo).
  */
  CImg<unsigned char> result_image = image;
  unsigned char red[] = {255, 0, 0};
  int threshold = 100; // Ajustable
  float _accu_h = ((sqrt(2.0) * (double)(h>w?h:w)) / 2.0) * 2; 
  std::vector<std::pair<int, int>> indices;
  for(int r=0;r<_accu_h;r++) {
    for(int t=0;t<180;t++) {
      if((int)cpuht[(r*180) + t] >= threshold) {
        indices.push_back(std::make_pair(r, t));
      }
    }
  }
  /*
   * Se calculan las coordenadas de dos puntos que forman una línea detectada en la imagen
  */
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

  /**
   * Se muestra y se guardan los resultados. 
   */
  result_image.display(); 
  result_image.save(outputFileName); 

  /*
    Se liberan los recursos.
  */
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_in);
  cudaFree(d_hough);
  free(h_hough);

  return 0;
}
