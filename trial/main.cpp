#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>

// Here is what i need to do to compile the program
// g++ main.cpp -o main `pkg-config --cflags --libs opencv`

using namespace cv;
using namespace std;


// Global Variables
const int slider_max = 10;
int slider;
Mat img;
// End of global variables

void on_trackbar(int position, void *)
{
    Mat img_converted;
    if (position > 0 )
        cvtColor(img, img_converted, CV_RGB2GRAY);
    else
        img_converted = img;
    imshow("Trackbar application", img_converted);
}



/**
 *  Description:    This method utilizes the laptop's camera and displays it
 *  Parameters:     None
 *  Return:
 *  Author:         Krishna Bhattarai
 *  Date:           November 8, 2015
 */
int display_video_from_webcam()
{
    // 0 is the ID of the built-in laptop camera, change if you want to use other camera
    VideoCapture my_stream(0);      // my_stream is a name i assigned

    //check if the file was opened properly
    if (!my_stream.isOpened())
    {
        cout << "Capture could not be opened successfully" << endl;
        return -1;
    }
    namedWindow("Krishna's Video");
    // Play the video in a loop till it ends
    while (char(waitKey(1)) != 'q' && my_stream.isOpened())
    {
        Mat frame;                      // declare a cvmat structure called frame
        my_stream >> frame;             // point the stream to frame
        // Check if the video is over
        if (frame.empty())
        {
            cout << "Video over" << endl;
            break;
        }
        imshow("Krishna's Video", frame);

    }


    return 0;
}


int gui_slider()
{
    img = imread("images.jpeg");
    namedWindow("Trackbar application");
    imshow("Trackbar application", img);
    slider = 0;

    createTrackbar("RGB <-> Grayscale", "Trackbar application", &slider, slider_max, on_trackbar);
    while (char(waitKey(1)) != 'q')
    {}
    return 0;
}




int apply_filter_matrix_to_an_image()
{
    Mat my_image = imread("images.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat filtered_image;
    // Filter kernel for detecting vertical edges
    float vertical_fk[5][5] = {{0,0,0,0,0}, {0,0,0,0,0}, {-1,-2,6,-2,-1}, {0,0,0,0,0}, {0,0,0,0,0}};
    float horizontal_fk[5][5] = {{0,0,-1,0,0}, {0,0,-2,0,0}, {0,0,6,0,0}, {0,0,-2,0,0}, {0,0,-1,0,0}};
    Mat filter_kernel_hor = Mat(5, 5, CV_32FC1, horizontal_fk);
    Mat filter_kernel_ver = Mat(5, 5, CV_32FC1, vertical_fk);

    // Apply filter

    filter2D(my_image, filtered_image, -1, filter_kernel_ver);

    namedWindow("Original");
    imshow("Original", my_image);

    namedWindow("Filtered image vertical");
    namedWindow("Filtered image horizontal");

    filter2D(my_image, filtered_image, -1, filter_kernel_hor);
    imshow("Filtered image vertical", filtered_image);
    imshow("Filtered image horizontal", filtered_image);
    while (char(waitKey(1)) != 'q')
    {}
    return 0;
}

int main()
{
    printf("Press 1 for video and 2 for image and 3 for image slider\n");
    printf("Please enter the appropriate key: ");
    char option;
    cin >> option;
    if (option == '1')
    {
        display_video_from_webcam();
    }
    if (option == '2')
    {
        apply_filter_matrix_to_an_image();
    }
    if (option == '3')
    {
        gui_slider();
    }
    return 0;
}