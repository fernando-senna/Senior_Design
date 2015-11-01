/**
 * Bhattarai, Krishna
 * Description:
 * Load an image using imread
 * Cerate a named OpenCv window using namedWindow
 * Display an image in an OpenCV window using imshow
 * http://docs.opencv.org/2.4/doc/tutorials/introduction/display_image/display_image.html
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Krishna's Image", WINDOW_AUTOSIZE );
    imshow("Krishna's Image", image);

    waitKey(0);

    return 0;
}
