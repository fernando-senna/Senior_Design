/*******************************************************************************************************************//**
* @file PupilTracker.cpp
* @brief Implementation for the PupilTracker class
*
* This class encapsulates the canny edge based pupil tracking algorithm
* @author Krishna Bhattarai
* @author Christopher D. McMurrough
***********************************************************************************************************************/

#include <math.h>                               // for standard math operations
#include <iostream>                             // for standard I/O
#include <stdio.h>                              // for standard I/O
#include <string>                               // for manipulating strings
#include <opencv2/core/core.hpp>                // Basic OpenCV structure (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>          // OpenCV window I/O
#include <opencv/highgui.h>                     // OpenCV window I/O
#include <opencv2/gpu/gpu.hpp>                  // Gpu Structures (gpu::GpuMat)
#include <time.h>                               // for timing things if needed
#include <opencv2/imgproc/imgproc.hpp>          // for basic image processing stuff like Gaussian Blur
#include "PupilTracker.h"                       // This is the lcoal pupil tracker class


/*******************************************************************************************************************//**
* @brief Constructor to create a PupilTracker
* @author Christopher D. McMurrough
***********************************************************************************************************************/
PupilTracker::PupilTracker()
{
    // initialize tracking processing variables
    m_courseDetection = false;
    m_coarse_filter_min = 100;
    m_coarse_filter_max = 700;

    m_blur = 1;
    m_canny_thresh = 300;
    m_canny_ratio = 2;
    m_canny_aperture = 5;

    // in the pupil tracker code intentisy filter is kept at 17
    // m_intensity_range = 17;
    m_intensity_range = 11;

    m_bin_thresh = 0;

    //int m_pupilIntensityOffset = 11;
    m_pupilIntensityOffset = 15;
    m_glintIntensityOffset = 5;

    // min coutour size is set at 60
    // m_min_contour_size = 60;
    m_min_contour_size = 80;

    m_inital_ellipse_fit_threshhold = static_cast<float>(1.8);
    m_min_ratio = 0.3f;
    m_pupil_min = 40.0f;
    m_pupil_max = 150.0f;
    m_target_size = 100.0f;
    m_strong_perimeter_ratio_range = cv::Point2f(0.8f, 1.1f);
    m_strong_area_ratio_range = cv::Point2f(0.6f, 1.1f);
    m_final_perimeter_ratio_range = cv::Point2f(0.6f, 1.2f);
    m_strong_prior = 0;

    m_confidence = 0;

    // debug settings
    m_display = false;
}

/*******************************************************************************************************************//**
* @brief Attempt to fit a pupil ellipse in the eye image frame
* @param[in] imageIn the input OpenCV image
* @param[out] result the output tracking data
* @return true if the a pupil was located in the image
* @author Christopher D. McMurrough
***********************************************************************************************************************/
bool PupilTracker::findPupil(const cv::gpu::GpuMat &imageIn)
{

    bool success = false;

    // Convert from bgr to grey
    cv::gpu::GpuMat imageGray, imageNormalized;
    /* gpu::cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn=0, Stream& stream=Stream::Null())*/
    cv::gpu::cvtColor(imageIn, imageGray, cv::COLOR_BGR2GRAY);

    // normalize the grey scale image
    const int rangeMin = 0;
    const int rangeMax = 255;
    //gpu::normalize(const GpuMat& src, GpuMat& dst, double alpha=1, double beta=0, int norm_type=NORM_L2, int dtype=-1, const GpuMat& mask=GpuMat())
    cv::gpu::normalize(imageGray, imageNormalized, rangeMin, rangeMax, cv::NORM_MINMAX, CV_8UC1);


    // Calculate the intensity histogram
//    cv::gpu::GpuMat histogram;
//    int histSize = 256;
//    // set the ranges that go from 0 to 255
//    float range[] = {0, 256}; // upper boundary is exclusive
//    const float *histRange = {range};

    /*void gpu::calcHist(const GpuMat& src, GpuMat& hist, Stream& stream=Stream::Null()) */
    /*void gpu::calcHist(const GpuMat& src, GpuMat& hist, GpuMat& buf, Stream& stream=Stream::Null()) */
    //cv::gpu::calcHist(&imageNormalized, &histogram,  );
    // ***************** Stuck at this point as I dont know what Stream is for **************

    // At this point I am going to assume that the histogram was calculated and I have the lowest and highest spike indices
    // A good idea would be to run the cpu code and print and see what these values would be
    // For now lets just go with default
    int lowest_spike_index = 255;
    int highest_spike_index = 0;
    float max_intensity = 0;

    // Create dark and spectral glint masks
    //cv::gpu::GpuMat binary_image, spectral_mask, kernel;
//    cv::Mat cpu_normal, cpu_spectral;
//    imageNormalized.upload(cpu_normal);
    //cv::inRange(cpu_normal, cv::InputArray(255), cv::InputArray(lowest_spike_index + m_pupilIntensityOffset), cpu_spectral);

    // Lets try to get the canny threshold
    cv::gpu::GpuMat edges;
    cv::gpu::Canny(imageNormalized, edges, m_canny_thresh, m_canny_thresh * m_canny_ratio, m_canny_aperture);
     if(m_display)
    {
        cv::Mat cpu_edges;
        edges.download(cpu_edges);
        cv::imshow("edges", cpu_edges);
    }

    return true;

    /**
    // compute the intensity histogram
    cv::Mat hist;
    int channels[] = {0};
    int histSize[] = {rangeMax - rangeMin + 1};
    float range[] = {static_cast<float>(rangeMin), static_cast<float>(rangeMax)};
    const float* ranges = {range};

    // find histogram spikes
    const int minSpikeSize = 40;
    int lowestSpike = rangeMax;
    int highestSpike = rangeMin;
    int numSpikes = 0;

    if(numSpikes < 2)
    {
        // not enough spikes, assign default values
        lowestSpike = 0;
        highestSpike = 255;
    }
    m_bin_thresh = lowestSpike;

    // create a mask for the dark pupil area (assign white to pupil area)
    cv::Mat darkMask;
    cv::inRange(imageGray, cv::InputArray(rangeMin), cv::InputArray(lowestSpike + m_pupilIntensityOffset), darkMask);
    const cv::Mat morphKernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::dilate(darkMask, darkMask, morphKernel, cv::Point(-1, -1), 2);
    if(m_display)
    {
        cv::imshow("darkMask", darkMask);
    }

    // create a mask for the light glint area (assign black to glint area)
    cv::Mat glintMask;
    cv::inRange(imageGray, cv::InputArray(rangeMin), cv::InputArray(highestSpike - m_glintIntensityOffset), glintMask);
    cv::erode(glintMask, glintMask, morphKernel, cv::Point(-1, -1), 1);
    if(m_display)
    {
        cv::imshow("glintMask", glintMask);
    }

    // line 141 to line 150 was commented out
    // remove eye lashes using an open morphology operation
    cv::Mat imageEyeLash;
    const cv::Mat openKernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(imageGray, imageEyeLash, cv::MORPH_OPEN, openKernel);
    if(m_display)
    {
        cv::imshow("eyeLash", imageEyeLash);
    }


    // apply additional blurring
    cv::Mat imageBlurred;
    if(m_blur > 1)
    {
        //cv::medianBlur(imageEyeLash, imageBlurred, m_blur);
        cv::medianBlur(imageGray, imageBlurred, m_blur);
    }
    else
    {
        //imageBlurred = imageEyeLash;
        imageBlurred = imageGray;
    }

    // compute canny edges
    cv::Mat edges;
    cv::Canny(imageBlurred, edges, m_canny_thresh, m_canny_thresh * m_canny_ratio, m_canny_aperture);
    if(m_display)
    {
        cv::imshow("edges", edges);
    }

    // remove edges outside of the white regions in the pupil and glint masks
    cv::Mat edgesPruned;
    cv::min(edges, darkMask, edgesPruned);
    cv::min(edgesPruned, glintMask, edgesPruned);

    if(m_display)
    {

        cv::imshow("edgesPruned", edgesPruned);
    }


    // calculate the centroid


    //m_ellipseCentroid = cv::Point2f(240, 240);
    cv::RotatedRect my_rotated_rect_property;

    my_rotated_rect_property = PupilTracker::getEllipseCentroid(edgesPruned);
    //m_ellipseRectangle = cv::RotatedRect(m_ellipseCentroid, cv::Size(20, 25), 0);
    // this is where it crashes
    if (std::isnan(my_rotated_rect_property.center.x) || std::isnan(my_rotated_rect_property.center.y)  || std::isnan(my_rotated_rect_property.size.width) || std::isnan(my_rotated_rect_property.size.height) )
    {
        success = false;
        return success;
    }
    else
    {
        success = true;
        m_ellipseRectangle = cv::RotatedRect(my_rotated_rect_property.center, my_rotated_rect_property.size, my_rotated_rect_property.angle);
    }

    m_crCenter = cv::Point2f(240, 240);
    m_crRadius = 1.0;
*/
    return true;

    // store the tracking result
//    if(success)
//    {
//        //m_ellipseCentroid = cv::Point2f(240, 240);
//        cv::RotatedRect my_rotated_rect_property;
//
//        my_rotated_rect_property = PupilTracker::getEllipseCentroid(edgesPruned);
//        //m_ellipseRectangle = cv::RotatedRect(m_ellipseCentroid, cv::Size(20, 25), 0);
//        if (my_rotated_rect_property.center.x == 0 && my_rotated_rect_property.center.y == 0 && my_rotated_rect_property.size.width == 0 && my_rotated_rect_property.size.height == 0 )
//        {
//            return false;
//        }
//        m_ellipseRectangle = cv::RotatedRect(my_rotated_rect_property.center, my_rotated_rect_property.size, my_rotated_rect_property.angle);
//        m_crCenter = cv::Point2f(240, 240);
//        m_crRadius = 1.0;
//
//        return true;
//    }

}

/*******************************************************************************************************************//**
* @brief Returns the pupil centroid
* @return pupil center as cv::Point2f
* @author Krishna Bhattarai
***********************************************************************************************************************/

//cv::Point2f PupilTracker::getEllipseCentroid(const cv::Mat &mask)
//{
//    cv::Moments m = moments(mask, true);
//    cv::Point center(m.m10/m.m00, m.m01/m.m00);
//    return center;
//}
cv::RotatedRect PupilTracker::getEllipseCentroid(const cv::Mat &mask) {
    //Even though it says centroid right now it is trying to return stuff needed for the
    // rotated rectangle.

    cv::Moments m = moments(mask, true);
    cv::RotatedRect ret;
    ret.center.x = m.m10 / m.m00;
    ret.center.y = m.m01 / m.m00;
    double mu20 = m.m20 / m.m00 - ret.center.x * ret.center.x;
    double mu02 = m.m02 / m.m00 - ret.center.y * ret.center.y;
    double mu11 = m.m11 / m.m00 - ret.center.x * ret.center.y;

    double common = std::sqrt(pow((mu20 - mu02), 2) + 4 * pow((mu11), 2));
    ret.size.width = std::sqrt(2 * (mu20 + mu02 + common));
    ret.size.height = std::sqrt(2 * (mu20 + mu02 - common));

    double num, den;
    if (mu02 > mu20) {
        num = mu02 - mu20 + common;
        den = 2 * mu11;

    } else {
        num = 2 * mu11;
        den = mu20 - mu02 + common;
    }

    if (num == 0 && den == 0)
        ret.angle = 0;
    else
        ret.angle = (180 / M_PI) * std::atan2(num, den);

    // The crash occurs when the values change to 0/0 for center, box.width, and box.height
    if (m_display)
    {
        std::cout << "center x: " << ret.center.x << " center y: " << ret.center.y << std::endl;
        std::cout << "box.width: " << ret.size.width << " box.height: " << ret.size.height << std::endl;
        std::cout << "rect.angle: " << ret.angle << "\n" << std::endl;
    }
    return ret;

//    cv::Point center(m.m10/m.m00, m.m01/m.m00);
//    return center;
}

cv::RotatedRect PupilTracker::getEllipseRectangle()
{
    return m_ellipseRectangle;
}
////
//cv::RotatedRect singleeyefitter::cvx::fitEllipse(const cv::Moments& m)
//{
//    using namespace math;
//    cv::RotatedRect ret;
//    ret.center.x = m.m10 / m.m00;
//    ret.center.y = m.m01 / m.m00;
//    double mu20 = m.m20 / m.m00 - ret.center.x * ret.center.x;
//    double mu02 = m.m02 / m.m00 - ret.center.y * ret.center.y;
//    double mu11 = m.m11 / m.m00 - ret.center.x * ret.center.y;
//    double common = std::sqrt(sq(mu20 - mu02) + 4 * sq(mu11));
//    ret.size.width = std::sqrt(2 * (mu20 + mu02 + common));
//    ret.size.height = std::sqrt(2 * (mu20 + mu02 - common));
//    double num, den;
//
//    if (mu02 > mu20) {
//        num = mu02 - mu20 + common;
//        den = 2 * mu11;
//
//    } else {
//        num = 2 * mu11;
//        den = mu20 - mu02 + common;
//    }
//
//    if (num == 0 && den == 0)
//        ret.angle = 0;
//    else
//        ret.angle = (180 / PI) * std::atan2(num, den);
//
//    return ret;
//}
/*******************************************************************************************************************//**
* @brief Returns the pupil centroid
* @return pupil center as cv::Point2f
* @author Christopher D. McMurrough
***********************************************************************************************************************/
// cv::Point2f PupilTracker::getEllipseCentroid()
// {
//     return m_ellipseCentroid;
// }

/*******************************************************************************************************************//**
* @brief Returns the pupil ellipse rectangle
* @return pupil ellipse rectangle as cv::RotatedRect
* @author Christopher D. McMurrough
***********************************************************************************************************************/


/*******************************************************************************************************************//**
* @brief Sets the display mode for the pupil tracker
* @author Christopher D. McMurrough
***********************************************************************************************************************/
void PupilTracker::setDisplay(bool display)
{
    m_display = display;
}


// This method calculates the highest and lowest intensities
void PupilTracker::calculate_spike_indices_and_max_intenesity(cv::gpu::GpuMat &histogram, int &amount_intensity_values,
  int& lowest_spike_index, int& highest_spike_index, float& max_intensity)
    {
        lowest_spike_index = 255;
        highest_spike_index = 0;
        max_intensity = 0;
        bool found_one = false;

        for (int i = 0; i < histogram.rows; i++) {
            //const float intensity  = histogram.at<float>(i, 0);
            // Since I cant use the term at i will chose 0.0f as a constant float value
            const float intensity = 0.0f;
            //  every intensity seen in more than amount_intensity_values pixels
            if (intensity > amount_intensity_values) {
                max_intensity = std::max(intensity, max_intensity);
                lowest_spike_index = std::min(lowest_spike_index, i);
                highest_spike_index = std::max(highest_spike_index, i);
                found_one = true;
            }
        }

        if (!found_one) {
            lowest_spike_index = 200;
            highest_spike_index = 255;
        }
    }

//cv::Vec2f singleeyefitter::cvx::majorAxis(const cv::RotatedRect& ellipse)
//{
//    return cv::Vec2f(ellipse.size.width * std::cos(PI / 180 * ellipse.angle), ellipse.size.width * std::sin(PI / 180 * ellipse.angle));
//}
