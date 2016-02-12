/*******************************************************************************************************************//**
* @file PupilTracker.cpp
* @brief Implementation for the PupilTracker class
*
* This class encapsulates the canny edge based pupil tracking algorithm
*
* @author Christopher D. McMurrough
***********************************************************************************************************************/

#include "PupilTracker.h"
#include "opencv2/opencv.hpp"
#include <iostream>

/*******************************************************************************************************************//**
* @brief Constructor to create a PupilTracker
* @author Christopher D. McMurrough
***********************************************************************************************************************/
PupilTracker::PupilTracker()
{
    // initialize tracking processing variables
    m_courseDetection = false;
    m_coarse_filter_min = 100;
    m_coarse_filter_max = 400;

    m_blur = 1;
    m_canny_thresh = 159;
    m_canny_ratio = 2;
    m_canny_aperture = 5;

    m_intensity_range = 11;
    m_bin_thresh = 0;

    m_pupilIntensityOffset = 11;
    //m_pupilIntensityOffset = 15;
    m_glintIntensityOffset = 5;

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
bool PupilTracker::findPupil(const cv::Mat& imageIn)
{
    bool success = true;

    // get the normalized grayscale image
    const int rangeMin = 0;
    const int rangeMax = 255;
    cv::Mat imageGray;
    cv::cvtColor(imageIn, imageGray, cv::COLOR_BGR2GRAY);
    cv::normalize(imageGray, imageGray, rangeMin, rangeMax, cv::NORM_MINMAX, CV_8UC1);
    if(m_display)
    {
        cv::imshow("imageGray", imageGray);
    }

    // compute the intensity histogram
    cv::Mat hist;
    int channels[] = {0};
    int histSize[] = {rangeMax - rangeMin + 1};
    float range[] = {static_cast<float>(rangeMin), static_cast<float>(rangeMax)};
    const float* ranges = {range};
    cv::calcHist(&imageGray, 1, channels, cv::Mat(), hist, 1, histSize, &ranges, true, false);

    // find histogram spikes
    const int minSpikeSize = 40;
    int lowestSpike = rangeMax;
    int highestSpike = rangeMin;
    int numSpikes = 0;
    for(int i = 0; i < histSize[0]; i++)
    {
        // check to see if we have a spike
        if(hist.at<uchar>(0, i) >= minSpikeSize)
        {
            numSpikes++;
            if(i < lowestSpike)
            {
                lowestSpike = i;
            }
            if(i > highestSpike)
            {
                highestSpike = i;
            }
        }
    }
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

// TODO: Add connected component analysis at this point to remove extra glint blobs

    // create a mask for the light glint area (assign black to glint area)
    cv::Mat glintMask;
    cv::inRange(imageGray, cv::InputArray(rangeMin), cv::InputArray(highestSpike - m_glintIntensityOffset), glintMask);
    cv::erode(glintMask, glintMask, morphKernel, cv::Point(-1, -1), 1);
    if(m_display)
    {
        cv::imshow("glintMask", glintMask);
    }

    /*
    // remove eye lashes using an open morphology operation
    cv::Mat imageEyeLash;
    const cv::Mat openKernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(imageGray, imageEyeLash, cv::MORPH_OPEN, openKernel);
    if(m_display)
    {
        cv::imshow("eyeLash", imageEyeLash);
    }
    */

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

    // compute the connected components out of the pupil edge candidates
    cv::Mat connectedEdges = cv::Mat::zeros(edgesPruned.size(), CV_8UC1);
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(edgesPruned, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    bool retryContourMerge = true;
    int relaxContourMerge = 0;
    while(retryContourMerge && contours.size() > 0)
    {
        for(int i = 0; i < contours.size(); i++)
        {
            // merge the contour if sufficiently large
            if(contours.at(i).size() < m_min_contour_size - relaxContourMerge)
            {
                continue;
            }
            else
            {
                cv::drawContours(connectedEdges, contours, i, cv::Scalar(255));
                retryContourMerge = false;
            }
        }
        relaxContourMerge += 2;
    }
    if(m_display)
    {
        cv::imshow("connectedEdges", connectedEdges);
    }
    
    // perform ellipse fitting
    cv::Moments m = cv::moments(connectedEdges, true);
    cv::RotatedRect rect;
    rect.center.x = m.m10 / m.m00;
    rect.center.y = m.m01 / m.m00;
    double mu20 = m.m20 / m.m00 - rect.center.x * rect.center.x;
    double mu02 = m.m02 / m.m00 - rect.center.y * rect.center.y;
    double mu11 = m.m11 / m.m00 - rect.center.x * rect.center.y;
    double common = std::sqrt(pow((mu20 - mu02), 2) + 4 * pow((mu11), 2));
    rect.size.width = std::sqrt(2 * (mu20 + mu02 + common));
    rect.size.height = std::sqrt(2 * (mu20 + mu02 - common));
    double num, den;
    if (mu02 > mu20) 
    {
        num = mu02 - mu20 + common;
        den = 2 * mu11;
    } 
    else
    {
        num = 2 * mu11;
        den = mu20 - mu02 + common;
    }
    if(num == 0 && den == 0)
    {
        rect.angle = 0;
    }
    else
    {
        rect.angle = (180 / m_pi) * std::atan2(num, den);   
    }

    // store the tracking result
    if(success)
    {
        m_ellipseCentroid = cv::Point2f(240, 240);
        m_ellipseRectangle = cv::RotatedRect(cv::Point2f(240, 240), cv::Size(10, 20), 0);
        m_ellipseRectangle = rect;
        m_crCenter = cv::Point2f(240, 240);
        m_crRadius = 1.0;

        return true;
    }
    else
    {
        // return false if tracking was not successful
        return false;
    }
}

/*******************************************************************************************************************//**
* @brief Returns the pupil centroid
* @return pupil center as cv::Point2f
* @author Christopher D. McMurrough
***********************************************************************************************************************/
cv::Point2f PupilTracker::getEllipseCentroid()
{
    return m_ellipseCentroid;
}

/*******************************************************************************************************************//**
* @brief Returns the pupil ellipse rectangle
* @return pupil ellipse rectangle as cv::RotatedRect
* @author Christopher D. McMurrough
***********************************************************************************************************************/
cv::RotatedRect PupilTracker::getEllipseRectangle()
{
    return m_ellipseRectangle;
}

/*******************************************************************************************************************//**
* @brief Sets the display mode for the pupil tracker
* @author Christopher D. McMurrough
***********************************************************************************************************************/
void PupilTracker::setDisplay(bool display)
{
    m_display = display;
}
