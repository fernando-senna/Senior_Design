/*******************************************************************************************************************//**
* @brief Ellipse class function implementaion file
* @author Krishna Bhattarai
***********************************************************************************************************************/

#include "Ellipse.h"
#include <cmath>
# define _USE_MATH_DEFINES
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
/*******************************************************************************************************************//**
* @brief Attempt to fit a pupil ellipse in the eye image frame
* @param[in] imageIn the input OpenCV image
* @param[out] result the output tracking data
* @return true if the a pupil was located in the image
* @author Christopher D. McMurrough
* @author Krishna Bhattarai
***********************************************************************************************************************/

Ellipse::Ellipse()
{
	 
}

cv::Point2f Ellipse::getEllipseCenter(const cv::RotatedRect& ellipse)
{
	return ellipse.center;
}

cv::Point2f Ellipse::getEllipseRadii(const cv::RotatedRect& ellipse)
{
	return ellipse.size;
}

double Ellipse::getEllipseAngle(const cv::RotatedRect& ellipse)
{
	return ellipse.angle;
}

double Ellipse::getEllipseArea(const cv::RotatedRect& ellipse)
{
	return M_PI * ellipse.size.width * ellipse.size.height;
}

double Ellipse::getEllipseCircumference(const cv::RotatedRect& ellipse)
{
	// using std::math
	double a = ellipse.size.width;
	double b = ellipse.size.height;

	double h = (a*a - b*b) / ((a +b) * (a+b));
	double circumference = M_PI * (a + b) * (1 + (3*h)/ (10 + sqrt(4 - 3*h)));
	return circumference;
}

// double getEllipseRatio(const cv::RotatedRect& ellipse);

double Ellipse::getEllipseMajorRadius(const cv::RotatedRect& ellipse)
{
	return ellipse.size.width;
}
double Ellipse::getEllipseMinorRadius(const cv::RotatedRect& ellipse)
{
	return ellipse.size.height;

}

double Ellipse::getEllipseCenterX(const cv::RotatedRect& ellipse)
{
	return ellipse.center.x;
}
double Ellipse::getEllipseCenterY(const cv::RotatedRect& ellipse)
{
	return ellipse.center.y;
}