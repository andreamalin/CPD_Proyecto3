#include <iostream>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "pgm.h"
#include <windows.h>
#include "CImg.h"
using namespace cimg_library;

 
#define M_PI 3.14159265358979323846
#define M_PI_2 (M_PI/2)
const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

// CPU function for Hough Transform
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    *acc = new int[rBins * degreeBins];
    memset(*acc, 0, sizeof(int) * rBins * degreeBins);
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            int idx = j * w + i;
            if (pic[idx] > 0) {
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                    float theta = tIdx * radInc;
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++;
                }
            }
        }
    }
}

// GPU kernel for Hough Transform
__global__ void GPU_HoughTran(unsigned char* pic, int w, int h, int* acc, float rMax, float rScale)
{
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID > w * h)
        return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;
    const float radInc = degreeInc * M_PI / 180;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * cosf(tIdx * radInc) + yCoord * sinf(tIdx * radInc);
            int rIdx = (r + rMax) / rScale;
            atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
        }
    }
}

// Function to draw a line on the image
void drawLine(unsigned char* img, int width, int height, int x1, int y1, int x2, int y2, unsigned char value) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            img[y1 * width + x1] = value;
        }

        if (x1 == x2 && y1 == y2)
            break;

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " input_image.bmp output_image.bmp" << std::endl;
        return 1;
    }

    const char* inputFileName = argv[1];
    const char* outputFileName = argv[2];

    CImg<unsigned char> image(inputFileName);
    if (image.is_empty()) {
        std::cout << "Failed to read image: " << inputFileName << std::endl;
        return 1;
    }

    int width = image.width();
    int height = image.height();
    // Allocate memory for accumulator array
    int* acc;
    cudaMallocManaged(&acc, sizeof(int) * rBins * degreeBins);
    memset(acc, 0, sizeof(int) * rBins * degreeBins);

    CPU_HoughTran(image.data(), width, height, &acc);

    // Find maximum value in accumulator array
    int maxVal = 0;
    for (int i = 0; i < rBins * degreeBins; i++) {
        if (acc[i] > maxVal) {
            maxVal = acc[i];
        }
    }

    // Threshold value for line detection
    int threshold = maxVal * 0.5;
    printf("%d\n", threshold);

    float xCent = width / 2.0f;
    float yCent = height / 2.0f;
    float rMax = sqrt(1.0f * width * width + 1.0f * height * height) / 2.0f;

    // Draw lines on the image
    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            if (acc[rIdx * degreeBins + tIdx] > threshold) {
                int r = rIdx * 2 - rBins;
                float theta = tIdx * radInc;

                // Calculate line endpoints in image coordinates
                int x1, y1, x2, y2;
                if (fabs(theta) < 1e-6 || fabs(theta - M_PI) < 1e-6) {
                    // Handle special cases where theta is close to 0 or π
                    x1 = x2 = static_cast<int>(round(xCent + r * cos(theta)));
                    y1 = 0;
                    y2 = height - 1;
                } else if (fabs(theta - M_PI_2) < 1e-6 || fabs(theta + M_PI_2) < 1e-6) {
                    // Handle special cases where theta is close to ±π/2
                    y1 = y2 = static_cast<int>(round(yCent - r * sin(theta)));
                    x1 = 0;
                    x2 = width - 1;
                } else {
                    x1 = static_cast<int>(round(xCent + (r - rMax * cos(theta))));
                    y1 = static_cast<int>(round(yCent - (rMax * sin(theta) + r)));
                    x2 = static_cast<int>(round(xCent + (r + rMax * cos(theta))));
                    y2 = static_cast<int>(round(yCent - (rMax * sin(theta) - r)));
                }

                drawLine(image.data(), width, height, x1, y1, x2, y2, 255);
            }
        }
    }

    // Save the output image
    image.save(outputFileName);
    // Clean up memory
    cudaFree(acc);

    return 0;
}
