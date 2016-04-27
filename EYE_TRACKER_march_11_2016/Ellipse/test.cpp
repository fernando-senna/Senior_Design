
#include <iostream>
#include <stdio.h>
#include <time.h>
#include "Ellipse.h"
//#include "Ellipse.cpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv)
{
	cv::RotatedRect my_ellipse;
	my_ellipse.center.x = 5;
	my_ellipse.center.x = 5;

	my_ellipse.size.width = 6;
	my_ellipse.size.height = 4;

	my_ellipse.angle = 40;

	Ellipse ellipse_class_instance;

	double area = ellipse_class_instance.getEllipseArea(my_ellipse);
	double circumference = ellipse_class_instance.getEllipseCircumference(my_ellipse);

	std::cout << "area, circumference" << area << circumference << std::endl;
	return 0;

}