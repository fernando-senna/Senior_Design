#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

#define NUM_COMNMAND_LINE_ARGUMENTS 2
#define CAMERA_FRAME_WIDTH 640
#define CAMERA_FRAME_HEIGHT 360
#define CAMERA_FORMAT CV_8UC1
#define CAMERA_FPS 30
#define CAMERA_BRIGHTNESS 128
#define CAMERA_CONTRAST 10
#define CAMERA_SATURATION 0
#define CAMERA_HUE 0
#define CAMERA_GAIN 0
#define CAMERA_EXPOSURE -6
#define CAMERA_CONVERT_RGB false

// color constants
CvScalar COLOR_WHITE = CV_RGB(255, 255, 255);
CvScalar COLOR_RED = CV_RGB(255, 0, 0);
CvScalar COLOR_GREEN = CV_RGB(0, 255, 0);
CvScalar COLOR_BLUE = CV_RGB(0, 0, 255);
CvScalar COLOR_YELLOW = CV_RGB(255, 255, 0);
CvScalar COLOR_MAGENTA = CV_RGB(255, 0, 255);


int main (int argc, char** argv)
{
// validate and parse the command line arguments

   
    return 0;
}
