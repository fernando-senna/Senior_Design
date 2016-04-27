
#ifndef ELLIPSE_H
#define ELLIPSE_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
class Ellipse
{
private:
    cv::RotatedRect my_ellipse;
    // define ellipse variables
    // cv::Point2f center; // (h, k)
    // each individual component of the center
    // float center_x;     // h
    // float center_y;     // k
    
    // cv::Point2f radii;  // (major, minor)
    // each individual component of the radius
    // float major_radius; // major or width
    // float mionr_radius; // minor or height

    // ellipse orientation angle
    // float angle;        // alpha
public:

    // constructors
    Ellipse();
    Ellipse(const cv::RotatedRect& ellipse);

    // accessors
    cv::Point2f getEllipseCenter(const cv::RotatedRect& ellipse);
    cv::Point2f getEllipseRadii(const cv::RotatedRect& ellipse);
    double getEllipseAngle(const cv::RotatedRect& ellipse);


    double getEllipseArea(const cv::RotatedRect& ellipse);
    double getEllipseCircumference(const cv::RotatedRect& ellipse);
    // double getEllipseRatio(const cv::RotatedRect& ellipse);

    double getEllipseMajorRadius(const cv::RotatedRect& ellipse);
    double getEllipseMinorRadius(const cv::RotatedRect& ellipse);

    double getEllipseCenterX(const cv::RotatedRect& ellipse);
    double getEllipseCenterY(const cv::RotatedRect& ellipse);
    // utility functions
    
};
#endif
 