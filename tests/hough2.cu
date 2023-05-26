#include <iostream>
#include <cmath>
#include <vector>
#include <CImg.h>

const double PI = 3.14159265358979323846;

struct Point {
    int x;
    int y;
};

std::vector<std::vector<int>> houghTransform(const std::vector<Point>& points, int width, int height, int threshold) {
    int diagonal = std::sqrt(width * width + height * height);
    int numThetas = 180;
    std::vector<std::vector<int>> accumulator(2 * diagonal, std::vector<int>(numThetas, 0));

    for (const auto& point : points) {
        int x = point.x;
        int y = point.y;

        for (int thetaIdx = 0; thetaIdx < numThetas; ++thetaIdx) {
            double theta = thetaIdx * PI / 180.0;
            int rho = std::round(x * std::cos(theta) + y * std::sin(theta));
            rho += diagonal;
            ++accumulator[rho][thetaIdx];
        }
    }

    std::vector<std::vector<int>> lines;
    for (int rho = 0; rho < 2 * diagonal; ++rho) {
        for (int thetaIdx = 0; thetaIdx < numThetas; ++thetaIdx) {
            if (accumulator[rho][thetaIdx] > threshold) {
                int rhoVal = rho - diagonal;
                lines.push_back({ rhoVal, thetaIdx });
            }
        }
    }

    return lines;
}

int main(int argc, char** argv) {
    const char* imagePath = "cuadrosHough.bmp";
    unsigned char thresholdValue = 128;
    unsigned char lineColor[] = { 255, 0, 0 }; // Red color (RGB values)

    // Load the image using CImg
    cimg_library::CImg<unsigned char> image(imagePath);
    cimg_library::CImg<unsigned char> binaryImage = image;
    binaryImage.threshold(thresholdValue);

    // Find contours in the binary image
    cimg_library::CImgList<cimg_library::CImg<int>> contours = binaryImage.get_isolines();

    // Draw the lines on the original image
    cimg_library::CImg<unsigned char> resultImage = image;
    // Estimate bounding boxes for the contours
    std::vector<cimg_library::CImg<int>> boundingBoxes;
    for (const auto& contour : contours) {
        int min_x = contour.width();
        int min_y = contour.height();
        int max_x = 0;
        int max_y = 0;

        cimg_forXY(contour, x, y) {
            if (contour(x, y) > 0) {
                if (x < min_x) min_x = x;
                if (y < min_y) min_y = y;
                if (x > max_x) max_x = x;
                if (y > max_y) max_y = y;
            }
        }

        // Convert the bounding box to unsigned char type
        cimg_library::CImg<unsigned char> bbox(resultImage.width(), resultImage.height(), 1, 3, 0);
        bbox.draw_line(min_x, min_y, max_x, min_y, lineColor);
        bbox.draw_line(max_x, min_y, max_x, max_y, lineColor);
        bbox.draw_line(max_x, max_y, min_x, max_y, lineColor);
        bbox.draw_line(min_x, max_y, min_x, min_y, lineColor);
        boundingBoxes.emplace_back(std::move(bbox));
    }

    // Perform Hough transform for line detection
    std::vector<Point> points;
    for (const auto& box : boundingBoxes) {
        int min_x = box.width();
        int min_y = box.height();
        int max_x = 0;
        int max_y = 0;

        cimg_forXY(box, x, y) {
            if (box(x, y) > 0) {
                if (x < min_x) min_x = x;
                if (y < min_y) min_y = y;
                if (x > max_x) max_x = x;
                if (y > max_y) max_y = y;
            }
        }

        // Extract the corner points of the bounding box
        points.push_back({ min_x, min_y });
        points.push_back({ max_x, min_y });
        points.push_back({ min_x, max_y });
        points.push_back({ max_x, max_y });
    }

    int width = image.width();
    int height = image.height();
    int threshold = 50; // Adjust the threshold value as needed
    std::vector<std::vector<int>> lines = houghTransform(points, width, height, threshold);

    int minLength = std::min(image.width(), image.height()) / 2; // Minimum line length

    for (const auto& line : lines) {
        int rho = line[0];
        int thetaIdx = line[1];
        double theta = thetaIdx * PI / 180.0;

        // Calculate the endpoints of the line
        double x0 = rho * std::cos(theta);
        double y0 = rho * std::sin(theta);
        double x1 = x0 + minLength * (-std::sin(theta));
        double y1 = y0 + minLength * (std::cos(theta));
        double x2 = x0 - minLength * (-std::sin(theta));
        double y2 = y0 - minLength * (std::cos(theta));

        // Draw the line on the original image
        resultImage.draw_line((int)x1, (int)y1, (int)x2, (int)y2, lineColor);
    }

    // Display the result image
    cimg_library::CImgDisplay display(resultImage, "Square and Line Detection Result");
    while (!display.is_closed()) {
        display.wait();
    }

    return 0;
}
