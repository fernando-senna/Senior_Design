#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static const std::string OPENCV_WINDOW = "Image window";

class ImageHandler
{
    ros::NodeHandle m_nodeHandle;
    image_transport::ImageTransport m_imageTransport;
    image_transport::Subscriber m_imageSubscriber;
    //image_transport::Publisher m_imagePublisher;

public:

    // constructor
    ImageHandler() : m_imageTransport(m_nodeHandle)
    {
        // subscribe to the image stream
        m_imageSubscriber = m_imageTransport.subscribe("/camera/image", 1, &ImageHandler::imageCallback, this);

        // publish an image stream
        //m_imagePublisher = m_imageTransport.advertise("/image_converter/output_video", 1);

        cv::namedWindow(OPENCV_WINDOW);
    }

    // destructor
    ~ImageHandler()
    {
        cv::destroyWindow(OPENCV_WINDOW);
    }

    // callback
    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cvPtr;

        // attempt to convert the image message to an openCV structure
        try
        {
            cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // PROCESS THE IMAGE HERE
        /*
        // Draw an example circle on the video stream
        if (cvPtr->image.rows > 60 && cvPtr->image.cols > 60)
            cv::circle(cvPtr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));
        */

        // display the image
        cv::imshow(OPENCV_WINDOW, cvPtr->image);
        cv::waitKey(3);

        // publish the result
        //m_imagePublisher.publish(cvPtr->toImageMsg());
    }
};

// program entry point
int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_converter");
    ImageHandler handler;
    ros::spin();

    return 0;
}
