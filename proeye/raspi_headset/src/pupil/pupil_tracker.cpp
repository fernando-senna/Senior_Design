/*******************************************************************************************************************//**
 * @FILE pupil_tracker.cpp
 * @BRIEF ROS warpper of the Robust Pupil Tracker by Lech Swirski
 *
 * Subscribes to video occulography ROS messages and performs pupil tracking using the Roboust Pupil Tracker by Lech
 * Swirski http://www.cl.cam.ac.uk/research/rainbow/projects/pupiltracking/.
 *
 * @AUTHOR Christopher D. McMurrough
 **********************************************************************************************************************/

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lib/PupilTracker.h"
#include "lib/cvx.h"

// camera parameters
#define CAMERA_FRAME_WIDTH 640
#define CAMERA_FRAME_HEIGHT 480

// define tracking parameters
#define MIN_RADIUS 10;
#define MAX_RADIUS 60
#define CANNY_BLUR 1.6
#define CANNY_THRESH_1 30
#define CANNY_THRESH_2 50
#define STARBURST_POINTS 0
#define PERCENT_INLIERS 40
#define INLIER_ITERATIONS 2
#define IMAGE_AWARE_SUPPORT true
#define EARLY_TERMINATION_PERCENTAGE 95
#define EARLY_REJECTION true
#define SEED_VALUE -1

// define color constants for image annotation
const CvScalar COLOR_WHITE = CV_RGB(255, 255, 255);
const CvScalar COLOR_RED = CV_RGB(255, 0, 0);
const CvScalar COLOR_GREEN = CV_RGB(0, 255, 0);
const CvScalar COLOR_BLUE = CV_RGB(0, 0, 255);
const CvScalar COLOR_YELLOW = CV_RGB(255, 255, 0);

// define a struct to hold tracking metadata
struct TrackingData
{
    float center_x;
    float center_y;
    float ellipse_angle;
    float ellipse_size;
    float cr_x;
    float cr_y;
    float cr_size;
    float dv_x;
    float dv_y;
};

// define the node and display window name
static const std::string NODE_NAME = "pupil tracking";
static const std::string DISPLAY_WINDOW_NAME = "pupil tracking";
std::string SUBSCRIBE_TOPIC_NAME = "/camera/image";

// define node settings
bool DISPLAY_RESULT = true;

// function prototypes
void processImage(cv::Mat &m, TrackingData &result);
void imageCallback();

/*******************************************************************************************************************//**
 * @BRIEF wrapper function for Roboust Pupil Tracker
 *
 * Perfoms a single iteration of pupil tracking on the input image frame
 *
 * @PARAM[in]  imageIn the input video occulography image
 * @PARAM[out] trackingResult the resulting tracking result metadata
 * @AUTHOR Christopher D. McMurrough
 **********************************************************************************************************************/
void processImage(cv::Mat &imageIn, PupilTracker::findPupilEllipse_out &trackingResult)
{
    // set the tracking parameters for this frame
    PupilTracker::TrackerParams params;
    params.Radius_Min = MIN_RADIUS;
    params.Radius_Max = MAX_RADIUS;
    params.CannyBlur = CANNY_BLUR;
    params.CannyThreshold1 = CANNY_THRESH_1;
    params.CannyThreshold2 = CANNY_THRESH_2;
    params.StarburstPoints = STARBURST_POINTS;
    params.PercentageInliers = PERCENT_INLIERS;
    params.InlierIterations = INLIER_ITERATIONS;
    params.ImageAwareSupport = IMAGE_AWARE_SUPPORT;
    params.EarlyTerminationPercentage = EARLY_TERMINATION_PERCENTAGE;
    params.EarlyRejection = EARLY_REJECTION;
    params.Seed = SEED_VALUE;

    // perform the ellipse fitting
    tracker_log log;
    PupilTracker::findPupilEllipse(params, imageIn, trackingResult, log);
}

/*******************************************************************************************************************//**
 * @BRIEF callback function for incoming video occulography images
 *
 * Handles image processing and display of annotated results
 *
 * @PARAM[in] msg input image message
 * @AUTHOR Christopher D. McMurrough
 **********************************************************************************************************************/
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cvPtr;
    PupilTracker::findPupilEllipse_out pupilResult;

    // attempt to convert the image message to an openCV structure
    try
    {
        cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // process the image
    processImage(cvPtr->image, pupilResult);

    // display the annotated result image if necessary
    if(DISPLAY_RESULT)
    {
        cvx::cross(cvPtr->image, pupilResult.pPupil, 5, COLOR_RED);
        cv::ellipse(cvPtr->image, pupilResult.elPupil, COLOR_GREEN);
        cv::imshow(DISPLAY_WINDOW_NAME, cvPtr->image);
        cv::waitKey(1);
    }

    // print the result
    ROS_INFO("Pupil center: %f %f", pupilResult.pPupil.x, pupilResult.pPupil.y);

    /*
    // package the result in the pupil data structure
    result->pupil_x = out.pPupil.x;
    result->pupil_y = out.pPupil.y;
    result->pupil_size = 0;
    result->cr_x = 0;
    result->cr_y = 0;
    result->cr_size = 0;
    result->dv_x = 0;
    result->dv_y = 0;
    */

    // publish the result
    //m_imagePublisher.publish(cvPtr->toImageMsg());
}

/*******************************************************************************************************************//**
 * @BRIEF program entry point
 *
 * Handles image processing and display of annotated results
 *
 * @PARAM[in] argc command line argument count
 * @PARAM[in] argv command line argument vector
 * @RETURNS return status
 * @AUTHOR Christopher D. McMurrough
 **********************************************************************************************************************/
int main(int argc, char** argv)
{
    // initialize the ROS node
    ros::init(argc, argv, "pupil_tracker");
    ros::NodeHandle nh;
    ros::NodeHandle n("~");

    // obtain parameters (don't forget to use a leading '_' when running via command line)
    n.param("topic", SUBSCRIBE_TOPIC_NAME, "/camera/image");
    n.param("display", DISPLAY_RESULT, true);

    // subscribe to the image stream
    image_transport::ImageTransport imageTransport(nh);
    image_transport::Subscriber imageSubscriber = imageTransport.subscribe(SUBSCRIBE_TOPIC_NAME, 1, imageCallback);

    // create the display window if necessasry
    if(DISPLAY_RESULT)
    {
        cv::namedWindow(DISPLAY_WINDOW_NAME);
    }

    // set the processing loop rate to 90 Hz
    ros::Rate loop_rate(90);

    // process data until program termination
    while(ros::ok())
    {
        // perform one iteration of message handling
        ros::spinOnce();

        // pause before sending the next update
        //loop_rate.sleep();
    }

    // release image structures
    if(DISPLAY_RESULT)
    {
        cv::destroyWindow(DISPLAY_WINDOW_NAME);
    }

    return 0;
}

