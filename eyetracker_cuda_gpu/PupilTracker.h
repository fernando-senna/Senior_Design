/**********************************************************************************************************************
* @file PupilTracker.cpp
* @brief Header for the PupilTracker class
*
* This class encapsulates the canny edge based pupil tracking algorithm
*
* @author Christopher D. McMurrough
***********************************************************************************************************************/

#ifndef PUPIL_TRACKER_H
#define PUPIL_TRACKER_H

#include <opencv2/core/core.hpp>

/**********************************************************************************************************************
* @class PupilTracker
*
* @brief Class for tracking pupils in an occulography image using canny edges
*
* The class implements a C++ version of the canny edge based pupil tracker by pupil-labs (originally written in Python)
* https://github.com/pupil-labs/pupil
*
* @author Christopher D. McMurrough
***********************************************************************************************************************/
class PupilTracker
{
private:

    // define tracking result parameters
    cv::Point2f m_ellipseCentroid;
    cv::RotatedRect m_ellipseRectangle;
    cv::Point2f m_crCenter;
    float m_crRadius;

    // define tracking processing variables
    bool m_courseDetection;
    int m_coarse_filter_min;
    int m_coarse_filter_max;

    int m_blur;
    int m_canny_thresh;
    int m_canny_ratio;
    int m_canny_aperture;

    int m_intensity_range;
    int m_bin_thresh;

    //int m_pupilIntensityOffse;
    int m_pupilIntensityOffset;
    int m_glintIntensityOffset;

    int m_min_contour_size;

    float m_inital_ellipse_fit_threshhold;
    float m_min_ratio;
    float m_pupil_min;
    float m_pupil_max;
    float m_target_size;
    cv::Point2f m_strong_perimeter_ratio_range;
    cv::Point2f m_strong_area_ratio_range;
    cv::Point2f m_final_perimeter_ratio_range;
    float m_strong_prior;

    float m_confidence;

    // debug settings
    bool m_display;

public:

    // constructors
    PupilTracker();

    // accessors
    cv::RotatedRect getEllipseCentroid(const cv::Mat &test);
    cv::RotatedRect getEllipseRectangle();

    // utility functions
    //bool findPupil(const cv::Mat& imageIn);
    bool findPupil(const cv::gpu::GpuMat &imageIn);
    void setDisplay(bool display);
};

#endif // PUPIL_TRACKER_H
