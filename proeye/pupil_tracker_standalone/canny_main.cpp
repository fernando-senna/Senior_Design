/*******************************************************************************************************************//**
 * @file main.cpp
 * @brief USB implementation of the canny pupil tracker
 *
 * USB implementation of the canny pupil tracker by pupil-labs
 * https://github.com/pupil-labs/pupil/
 *
 * @author Christopher D. McMurrough
 **********************************************************************************************************************/

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <opencv/highgui.h>

#include "canny_pupil/PupilTracker.h"

// configuration parameters
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
    //cv::VideoCapture occulography(cameraIndex);
   cv::VideoCapture occulography("pupil_test.mp4");
   // cv::VideoCapture occulography("test.mov");
    if(!occulography.isOpened())
    {
        std::printf("Unable to initialize camera %u! \n", cameraIndex);
        return 0;
    }
    occulography.set(CV_CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH);
    occulography.set(CV_CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT);
    occulography.set(CV_CAP_PROP_FORMAT, CAMERA_FORMAT);
    occulography.set(CV_CAP_PROP_FPS, CAMERA_FPS);
    occulography.set(CV_CAP_PROP_BRIGHTNESS, CAMERA_BRIGHTNESS);
    occulography.set(CV_CAP_PROP_CONTRAST, CAMERA_CONTRAST);
    occulography.set(CV_CAP_PROP_SATURATION, CAMERA_SATURATION);
    occulography.set(CV_CAP_PROP_HUE, CAMERA_HUE);
    occulography.set(CV_CAP_PROP_GAIN, CAMERA_GAIN);
    occulography.set(CV_CAP_PROP_EXPOSURE, CAMERA_EXPOSURE);
    occulography.set(CV_CAP_PROP_CONVERT_RGB, CAMERA_CONVERT_RGB);

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
                std::printf("Unable to locate pupil! \n");
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
                    cv::ellipse(displayImage, tracker.getEllipseRectangle(), COLOR_RED);

                    // shade the pupil area
                    cv::Mat annotation(eyeImage.rows, eyeImage.cols, CV_8UC3, 0.0);
                    cv::ellipse(annotation, tracker.getEllipseRectangle(), COLOR_MAGENTA, -1);
                    const double alpha = 0.7;
                    cv::addWeighted(displayImage, alpha, annotation, 1.0 - alpha, 0.0, displayImage);
                }

                if(flipDisplay)
                {
                    // annotate the image
                    cv::Mat displayFlipped;
                    cv::flip(displayImage, displayFlipped, 1);
                    cv::imshow("eyeImage", displayFlipped);

                    // display the annotated image
                    isRunning = cv::waitKey(1) != 'q';
                    displayFlipped.release();
                }
                else
                {
                    // display the image
                    cv::imshow("eyeImage", displayImage);
                    isRunning = cv::waitKey(1) != 'q';
                }

                // release display image
                displayImage.release();
            }
        }
        else
        {
            std::printf("WARNING: Unable to capture image from source!\n");
            occulography.set(CV_CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // stop the timer and print the elapsed time
        frameEndTicks = clock();
        totalTime = ((float)(frameEndTicks - frameStartTicks)) / CLOCKS_PER_SEC;
        std::printf("Processing time (pupil, total) (result x,y): %.4f %.4f - %.2f %.2f\n", processTime, totalTime, tracker.getEllipseCentroid().x, tracker.getEllipseCentroid().y);
    }

    // release the video source before exiting
    occulography.release();
}

