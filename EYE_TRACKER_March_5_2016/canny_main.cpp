/*******************************************************************************************************************//**
 * @file main.cpp
 * @brief USB implementation of the canny pupil tracker
 *
 * USB implementation of the canny pupil tracker by pupil-labs
 * https://github.com/pupil-labs/pupil/
 *
 * @author Christopher D. McMurrough
 * @author Krishna Bhattarai
 **********************************************************************************************************************/

#include <iostream>
#include <stdio.h>
#include <time.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <string>
#include "PupilTracker.h"

// configuration parameters
#define NUM_COMNMAND_LINE_ARGUMENTS 2
#define CAMERA_FRAME_WIDTH 640
#define CAMERA_FRAME_HEIGHT 360
#define CAMERA_FORMAT CV_8UC1
#define CAMERA_FPS 30

#define CAMERA_BRIGHTNESS -10
#define CAMERA_CONTRAST 0
#define CAMERA_SATURATION 0
#define CAMERA_HUE 0
#define CAMERA_GAIN 0
#define CAMERA_EXPOSURE -60
#define CAMERA_CONVERT_RGB false

// color constants
CvScalar COLOR_WHITE = CV_RGB(255, 255, 255);
CvScalar COLOR_RED = CV_RGB(255, 0, 0);
CvScalar COLOR_GREEN = CV_RGB(0, 255, 0);
CvScalar COLOR_BLUE = CV_RGB(0, 0, 255);
CvScalar COLOR_YELLOW = CV_RGB(255, 255, 0);
CvScalar COLOR_MAGENTA = CV_RGB(255, 0, 255);
CvScalar COLOR_SILVER = CV_RGB(192, 192, 192);
CvScalar COLOR_GRAY = CV_RGB(128, 128, 128);
CvScalar COLOR_PURPLE = CV_RGB(128, 0, 128);
CvScalar COLOR_CYAN = CV_RGB(0, 255, 255);
CvScalar COLOR_SPRING = CV_RGB(127, 255, 212);
CvScalar COLOR_LIGHTGREEN = CV_RGB(0,255,127);


/*******************************************************************************************************************//**
 * @brief Program entry point
 *
 * Handles image processing and display of annotated results
 *
 * @param[in] argc command line argument count
 * @param[in] argv command line argument vector
 * @returnS return status
 * @author Christopher D. McMurrough
 **********************************************************************************************************************/
int main(int argc, char** argv)
{
    // validate and parse the command line arguments
    int cameraIndex = 0;
    bool displayMode = true;
    bool flipDisplay = false;
    if(argc != NUM_COMNMAND_LINE_ARGUMENTS + 1)
    {
        std::printf("USAGE: <camera_index> <display_mode>\n");
        std::printf("Running with default parameters... \n");
    }
    else
    {
        cameraIndex = atoi(argv[1]);
        displayMode = atoi(argv[2]) > 0;
        flipDisplay = atoi(argv[2]) == 2;
    }

    // initialize the eye camera video capture
//    cv::VideoCapture occulography(cameraIndex);
    cv::VideoCapture occulography("pupil_test.mp4");

    if(!occulography.isOpened())
    {
        std::printf("Unable to initialize camera %u! \n", cameraIndex);
        return 0;
    }
    // THESE ARE FOR GETTING THE PROPERTIES OF THE CAMERA IF APPLICABLE

    std::cout<<"width "<< occulography.get(CV_CAP_PROP_FRAME_WIDTH)  << std::endl;
    std::cout<<"height "<< occulography.get(CV_CAP_PROP_FRAME_WIDTH)  << std::endl;

    

    // std::cout<<"format "<< occulography.get(CV_CAP_PROP_FORMAT)  << std::endl;
    // std::cout<<"fps "<< occulography.get(CV_CAP_PROP_FPS)  << std::endl;
    // std::cout<<"brightness "<< occulography.get(CV_CAP_PROP_BRIGHTNESS)  << std::endl;
    // std::cout<<"contrast "<< occulography.get(CV_CAP_PROP_CONTRAST)  << std::endl;
    // std::cout<<"saturation "<< occulography.get(CV_CAP_PROP_SATURATION)  << std::endl;
    // std::cout<<"hue "<< occulography.get(CV_CAP_PROP_HUE)  << std::endl;
    // std::cout<<"gain "<< occulography.get(CV_CAP_PROP_GAIN)  << std::endl;
    // std::cout<<"exposure "<< occulography.get(CV_CAP_PROP_EXPOSURE)  << std::endl;
    // std::cout<<"convert rgb "<< occulography.get(CV_CAP_PROP_CONVERT_RGB)  << std::endl;

    // THESE ARE FOR SETTING THE PROPERTIES OF THE CAMERA IF APPLICABLE
    occulography.set(CV_CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH);
    occulography.set(CV_CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT);
    // occulography.set(CV_CAP_PROP_FORMAT, CAMERA_FORMAT);
    // occulography.set(CV_CAP_PROP_FPS, CAMERA_FPS);
    occulography.set(CV_CAP_PROP_BRIGHTNESS, CAMERA_BRIGHTNESS);
    occulography.set(CV_CAP_PROP_CONTRAST, CAMERA_CONTRAST);
    occulography.set(CV_CAP_PROP_SATURATION, CAMERA_SATURATION);
    // occulography.set(CV_CAP_PROP_HUE, CAMERA_HUE);
    // occulography.set(CV_CAP_PROP_GAIN, CAMERA_GAIN);
    // occulography.set(CV_CAP_PROP_EXPOSURE, CAMERA_EXPOSURE);
    // occulography.set(CV_CAP_PROP_CONVERT_RGB, CAMERA_CONVERT_RGB);

    // intialize the display window if necessary
    if(displayMode)
    {
        cvNamedWindow("eyeImage", CV_WINDOW_NORMAL);
        cvSetWindowProperty("eyeImage", CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
        cvSetWindowProperty("eyeImage", CV_WND_PROP_AUTOSIZE, CV_WINDOW_NORMAL);
        cvSetWindowProperty("eyeImage", CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);
    }

    // create the pupil tracking object
    PupilTracker tracker;
    tracker.setDisplay(displayMode);

    // store the frame data
    cv::Mat eyeImage;
    //struct PupilData result;
    bool trackingSuccess = false;

    // store the time between frames
    int frameStartTicks, frameEndTicks, processStartTicks, processEndTicks;
    float processTime, totalTime;

    // process data until program termination
    bool isRunning = true;
    while(isRunning)
    {
        // start the timer
        frameStartTicks = clock();

        // attempt to acquire an image frame
        if(occulography.read(eyeImage))
        {
            // process the image frame
            processStartTicks = clock();
            trackingSuccess = tracker.findPupil(eyeImage);
            processEndTicks = clock();
            processTime = ((float)(processEndTicks - processStartTicks)) / CLOCKS_PER_SEC;

            // warn on tracking failure
            if(!trackingSuccess)
            {
                std::cout<< "Unable to locate pupil!" << std::endl;
            }

            // update the display
            if(displayMode)
            {
                cv::Mat displayImage(eyeImage);

                // annotate the image if tracking was successful
                if(trackingSuccess)
                {
                    // draw the pupil center and boundary
                    //cvx::cross(displayImage, result.pupil_center, 5, COLOR_RED);
                    cv::ellipse(displayImage, tracker.getEllipseRectangle(), COLOR_BLUE, 2);

                    // shade the pupil area
                    cv::Mat annotation(eyeImage.rows, eyeImage.cols, CV_8UC3, 0.0);
                    cv::RotatedRect my_rotated_rect_property;
                    my_rotated_rect_property = tracker.getEllipseRectangle();
                    cv::Point2f Center;
                    Center = my_rotated_rect_property.center;

                    double x1, y1, x2, y2;
                    
                    x1 = Center.x - 40; y1 = Center.y - 40; x2 = Center.x + 40; y2 = Center.y + 40;

                    // Draw a square/rectangle around the ellipse based off of the center 
                    // This is for the rectangle
                    cv::Point x1y1 = cv::Point(x1, y1);
                    cv::Point x2y2 = cv::Point(x2, y2);
                    // This is for the lines around the rectangle 
                    cv::Point x1y2 = cv::Point(x1, y2);
                    cv::Point x2y1 = cv::Point(x2, y1);

                    // here is the math to draw the lines 
                    // for the horizontal line
                    cv::Point horizontal_start = cv::Point(x1-20, (y1+y2)/2);
                    cv::Point horizontal_end = cv::Point(x1+100, (y1+y2)/2);
                    
                    // for the vertical line
                    cv::Point vertical_start = cv::Point((x1+x2)/2, y1-20);
                    cv::Point vertical_end = cv::Point((x1+x2)/2, y1+100);
                    
                    // done with the lines

                    cv::ellipse(annotation, tracker.getEllipseRectangle(), COLOR_PURPLE, -1);
                    const double alpha = 0.7;
                    cv::addWeighted(displayImage, alpha, annotation, 1.0 - alpha, 0.0, displayImage);
                    // The centre is actually a tiny circle (neat trick)
                    cv::circle(displayImage, Center, 2.5, COLOR_RED, 2, 4);
                    // The bounding rectangle based off of the centre
                    cv::rectangle(displayImage, x1y1, x2y2, COLOR_GREEN, 1);
                    // The lines across the rectangle
                    cv::line(displayImage, horizontal_start, horizontal_end, COLOR_GREEN, 1);
                    cv::line(displayImage, vertical_start, vertical_end, COLOR_GREEN, 1);
                    int fontFace = cv::FONT_HERSHEY_PLAIN;
                    double scale = 1.5;
                    int thickness = 1;

                    int baseline = 0;
                    char pupil_center[25];
                    char ellipse_size[25];
                    char pupil_angle[25];
                    sprintf(pupil_center, "Pupil Center: (%.2lf, %.2lf)",  Center.x, Center.y);
                    sprintf(ellipse_size, "Pupil Size: (%.2lf, %.2lf)",  my_rotated_rect_property.size.width,my_rotated_rect_property.size.height);
 					sprintf(pupil_angle, "Pupil Angle: (%.2lf)",  my_rotated_rect_property.angle);
                    cv::putText(displayImage, "OptiFind :: Real Time Eye Tracker", cv::Point(40, 40), fontFace, 1, COLOR_WHITE, thickness, 8);
                    cv::putText(displayImage, pupil_center, cv::Point(350, 315), fontFace, 1, COLOR_WHITE, thickness, 8);
                    cv::putText(displayImage, ellipse_size, cv::Point(350, 330), fontFace, 1, COLOR_WHITE, thickness, 8);
                    cv::putText(displayImage, pupil_angle, cv::Point(350, 345), fontFace, 1, COLOR_WHITE, thickness, 8);
                 }

                if(flipDisplay)
                {
                    // annotate the image
                    cv::Mat displayFlipped;
                    cv::flip(displayImage, displayFlipped, 1);
                    cv::imshow("eyeImage", displayFlipped);

                    // display the annotated image
                    isRunning = cv::waitKey(1) != 'q';
                    char key = cvWaitKey(10);
                    // If the user pressed 'ESC' Key then break the loop and exit the program
                    if (key == 27)
                        break;
                    displayFlipped.release();
                }
                else
                {
                    // display the image
                    cv::imshow("eyeImage", displayImage);
                    //char key1 = cvWaitKey(0);
                    isRunning = cv::waitKey(1) != 'q';
                    char key = cvWaitKey(10);
                    // If the user pressed 'ESC' Key then break the loop and exit the program
                    if (key == 27)
                        break;
                }
                // release display image
                displayImage.release();
            }
        }
        else
        {
            std::cout<<"WARNING: Unable to capture image from source!\n"<<std::endl;
            std::cout<< "Reading the video from frame 0 again!" << std::endl;
            occulography.set(CV_CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // stop the timer and print the elapsed time
        // frameEndTicks = clock();
        // totalTime = ((float)(frameEndTicks - frameStartTicks)) / CLOCKS_PER_SEC;
        // std::printf("Processing time (pupil, total) (result x,y): %.4f %.4f - %.2f %.2f\n", processTime, totalTime, tracker.getEllipseRectangle().center.x, tracker.getEllipseRectangle().center.y);
        //std::cout<< "Processing time (pupil)"<< processTime << "total time" <<  totalTime << "pupil center" << tracker.getEllipseRectangle().center << std::endl;
       
    }

    // release the video source before exiting
    occulography.release();
}
