/*******************************************************************************************************************//**
* @file PupilTracker.cpp
* @brief Implementation for the PupilTracker class
*
* This class encapsulates the canny edge based pupil tracking algorithm
*
* @author Christopher D. McMurrough
* @author Krishna Bhattarai
***********************************************************************************************************************/

#include "PupilTracker.h"
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <iostream>

typedef std::vector<std::vector<cv::Point> > Contours_2D;
typedef std::vector<cv::Point> Contour_2D;
typedef std::vector<cv::Point> Edges2D;
typedef std::vector<int> ContourIndices;

#define PI std::acos(-1);
#define TWO_PI = 2.0 * sd::acos(-1);
/*******************************************************************************************************************//**
* @brief Constructor to create a PupilTracker
* @author Christopher D. McMurrough
* @author Krishna Bhattarai
***********************************************************************************************************************/
PupilTracker::PupilTracker()
{
    // initialize tracking processing variables
    m_courseDetection = false;
    m_coarse_filter_min = 100;
    m_coarse_filter_max = 400;

    m_blur = 1;
    m_canny_thresh = 350;
    m_canny_ratio = 2;
    m_canny_aperture = 5;

    m_intensity_range = 17;
    m_bin_thresh = 0;

    m_pupilIntensityOffset = 11;
    m_glintIntensityOffset = 5;

    m_min_contour_size = 400;

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
* @author Krishna Bhattarai
***********************************************************************************************************************/
bool PupilTracker::findPupil(const cv::Mat& imageIn)
{
    bool success = false;

    // get the normalized grayscale image
    if(!m_display)
    {
        cv::imshow("infrared ", imageIn);
    }

    const int rangeMin = 0;
    const int rangeMax = 255;
    cv::Mat imageGray;
    cv::cvtColor(imageIn, imageGray, cv::COLOR_BGR2GRAY);
    cv::normalize(imageGray, imageGray, rangeMin, rangeMax, cv::NORM_MINMAX, CV_8UC1);
    if(!m_display)
    {
        cv::imshow("imageGray", imageGray);
    }

    // Alternative implementation of Histogram spikes
    cv::Mat histogram;
    int histSize;
    histSize = 256; //from 0 to 255
    /// Set the ranges
    float range[] = { 0, 256 } ; //the upper boundary is exclusive
    const float* histRange = { range };
    cv::calcHist(&imageGray, 1 , 0, cv::Mat(), histogram , 1 , &histSize, &histRange, true, false);

    int lowest_spike_index = 255;
    int highest_spike_index = 0;
    float max_intensity = 0;

    PupilTracker::calculate_spike_indices_and_max_intenesity(histogram, 40, lowest_spike_index, highest_spike_index, max_intensity);
    // Alternative implementation of histogram spikes ends here

    m_bin_thresh = lowest_spike_index;

    // create a mask for the dark pupil area (assign white to pupil area)
    cv::Mat darkMask;
    cv::inRange(imageGray, cv::InputArray(rangeMin), cv::InputArray(lowest_spike_index + m_pupilIntensityOffset), darkMask);
    const cv::Mat morphKernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::dilate(darkMask, darkMask, morphKernel, cv::Point(-1, -1), 2);
    if(!m_display)
    {
        cv::imshow("darkMask", darkMask);
    }

    // create a mask for the light glint area (assign black to glint area)
    cv::Mat glintMask;
    cv::inRange(imageGray, cv::InputArray(rangeMin), cv::InputArray(highest_spike_index - m_glintIntensityOffset), glintMask);
    cv::erode(glintMask, glintMask, morphKernel, cv::Point(-1, -1), 1);
    if(!m_display)
    {
        cv::imshow("glintMask", glintMask);
    }

    // remove eye lashes using an open morphology operation
    cv::Mat imageEyeLash;
    const cv::Mat openKernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(imageGray, imageEyeLash, cv::MORPH_OPEN, openKernel);
    if(!m_display)
    {
        cv::imshow("eyeLash", imageEyeLash);
    }
    

    // apply additional blurring
    cv::Mat imageBlurred;
    if(m_blur >= 1)
    {
        cv::medianBlur(imageEyeLash, imageBlurred, m_blur);
        //cv::medianBlur(imageGray, imageBlurred, m_blur);
    }
    else
    {
        //imageBlurred = imageEyeLash;
        imageBlurred = imageGray;
    }

    // compute canny edges
    cv::Mat edges;
    cv::Canny(imageBlurred, edges, m_canny_thresh, m_canny_thresh * m_canny_ratio, m_canny_aperture);
    if(!m_display)
    {
        cv::imshow("edges", edges);
    }

    // remove edges outside of the white regions in the pupil and glint masks
    cv::Mat edgesPruned;
    cv::min(edges, darkMask, edgesPruned);
    cv::min(edgesPruned, glintMask, edgesPruned);

   
    if(!m_display)
    {

        cv::imshow("edgesPruned", edgesPruned);
    }

    // compute the connected components out of the pupil edge candidates
    cv::Mat connectedEdges = cv::Mat::zeros(darkMask.size(), CV_8UC1);
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(darkMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    // Alternatives 
    // cv::findContours(darkMask, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    // cv::findContours(edgesPruned, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

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
  

    if(!m_display)
    {
        cv::imshow("connectedEdges", connectedEdges);
    }
   
    cv::RotatedRect my_rotated_rect_property;
    my_rotated_rect_property = PupilTracker::getEllipseCentroid(connectedEdges);
   
    // this is where it crashes if the ellipse properties are not checked 
    if (std::isnan(my_rotated_rect_property.center.x) || std::isnan(my_rotated_rect_property.center.y)  || std::isnan(my_rotated_rect_property.size.width) || std::isnan(my_rotated_rect_property.size.height) )
    {
        success = false;
        return success;
    }
    else
    {
        m_ellipseRectangle = cv::RotatedRect(my_rotated_rect_property.center, my_rotated_rect_property.size, my_rotated_rect_property.angle);
    }
    return true;
}


/*******************************************************************************************************************//**
* @brief This method utilizes moments to calculate the center of mass of a mask (dark or spectral)
* @param[in] cv::Mat &mask 
* @param[out] cv:: RotatedRect
* @return cv::RotatedRect 
* @author Krishna Bhattarai
***********************************************************************************************************************/

cv::RotatedRect PupilTracker::getEllipseCentroid(const cv::Mat &mask) 
{
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
    if (!m_display)
    {
        std::cout << "center x: " << ret.center.x << " center y: " << ret.center.y << std::endl;
        std::cout << "box.width: " << ret.size.width << " box.height: " << ret.size.height << std::endl;
        std::cout << "rect.angle: " << ret.angle << "\n" << std::endl;
    }
    return ret;
}


/*******************************************************************************************************************//**
* @brief Lets the main function grab the results of the pupils properties
* @author Christopher D. McMurrough
* @author Krishna Bhattarai
***********************************************************************************************************************/
cv::RotatedRect PupilTracker::getEllipseRectangle()
{
    return m_ellipseRectangle;
}

/*******************************************************************************************************************//**
* @brief Sets the display mode for the pupil tracker
* @author Christopher D. McMurrough
* @author Christopher D. McMurrough
***********************************************************************************************************************/
void PupilTracker::setDisplay(bool display)
{
    m_display = display;
}

/*******************************************************************************************************************//**
* @brief Calculates the histogram spikes and sets those values using their address
* @author Christopher D. McMurrough
* @author Krishna Bhattarai
***********************************************************************************************************************/
void PupilTracker::calculate_spike_indices_and_max_intenesity(cv::Mat& histogram,
                                                              int amount_intensity_values,
                                                              int& lowest_spike_index,
                                                              int& highest_spike_index,
                                                              float& max_intensity)
{
    lowest_spike_index = 255;
    highest_spike_index = 0;
    max_intensity = 0;
    bool found_one = false;

    for (int i = 0; i < histogram.rows; i++) {
        const float intensity  = histogram.at<float>(i, 0);

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

/*******************************************************************************************************************//**
* @brief calculates the major Axis of a rotatedrect object
* @author Krishna Bhattarai
***********************************************************************************************************************/
cv::Vec2f PupilTracker::majorAxis(const cv::RotatedRect& ellipse)
{
    return cv::Vec2f(ellipse.size.width * std::cos(3.14 / 180 * ellipse.angle), ellipse.size.width * std::sin(3.14 / 180 * ellipse.angle));
}

/*******************************************************************************************************************//**
* @brief An attempt to initialize a bounding box
* @author Krishna Bhattarai
***********************************************************************************************************************/
inline cv::Rect boundingBox(const cv::Mat& img)
{
    return cv::Rect(0, 0, img.cols, img.rows);
}

/*******************************************************************************************************************//**
* @brief This is what gets us the region of interest. Even though the parameter has something called roi, it is just an 
  empty container to hold the roi that we are going to compute. Later on we will overlay this roi with whatever image we want.
  Get the ROI where the boarder of black pixels is removed. This was an adaption from the pupil laps eye tracking code
* @author Krishna Bhattarai
***********************************************************************************************************************/
bool PupilTracker::getRoiWithoutBorder(const cv::Mat& img , cv::Rect& roi)
{

    CV_Assert(img.depth() == CV_8U);
    CV_Assert(img.isContinuous());

    if (img.total() == 0) return  false;

    int n_rows = img.rows, n_cols = img.cols;
    int x_min = 0, y_min = 0;
    int x_max = n_cols - 1, y_max = n_rows - 1;

    const uchar* img_ptr = img.data;
    bool found = false;
    bool break_loop = false;

    // find the roi where all pixles outside are zero
    // instead of iterating through the whole image
    // we try each side and find the first none zero point

    //from top, find the y where the first non-zero pixel occures in a row
    for (int i = 0; i < n_rows * n_cols; i++) {
        if (*img_ptr != 0) {
            int row = int(i / n_cols);
            y_min = row;
            found = true;
            break;
        }

        img_ptr++;
    } // end loop

    if (found == false) return  false; // we can stop here, nothing found

    // from bottom, find the y where the first non-zero pixel occures in a row
    img_ptr = &img.data[n_rows * n_cols - 1];

    for (int i = n_rows * n_cols - 1; i >= 0; i--) {
        if (*img_ptr != 0) {
            int row = int(i / n_cols);
            y_max = row;
            break;
        }

        img_ptr--;
    } // end loop

    // from left, find the x where the first non-zero pixel occures in a column
    // ignore y values lower or higher the one we already found
    for (int i = 0; i < n_cols; i++) {
        for (int j = y_min; j <= y_max; j++) {
            img_ptr = &img.data[i +  j * n_cols];

            if (*img_ptr != 0) {
                x_min = i;
                break_loop = true;
            }
        }

        if (break_loop) break;

    } // end loop

    break_loop = false;

    // // from right, find the x where the first non-zero pixel occures in a column
    // ignore y values lower or higher the one we already found
    for (int i = n_cols - 1; i >= 0 ; i--) {
        for (int j = y_min; j <= y_max; j++) {
            img_ptr = &img.data[i +  j * n_cols];

            if (*img_ptr != 0) {
                x_max = i;
                break_loop = true;
            }
        }

        if (break_loop) break;

    } // end loop

    roi =  cv::Rect(x_min, y_min, 1 + (x_max-x_min) , 1 + (y_max-y_min ) );
    return true;
}



/*******************************************************************************************************************//**
* @brief This is so that we can put a bounding box/ boarder around the roi
* This was an adaption from the pupil laps eye tracking code
* @author Krishna Bhattarai
***********************************************************************************************************************/
void PupilTracker::getROI(const cv::Mat& src, cv::Mat& dst, const cv::Rect& roi, int borderType)
{
    cv::Rect bbSrc = boundingBox(src);
    cv::Rect validROI = roi & bbSrc;

    if (validROI == roi) {
        dst = cv::Mat(src, validROI);

    } else {
        // Figure out how much to add on for top, left, right and bottom
        cv::Point tl = roi.tl() - bbSrc.tl();
        cv::Point br = roi.br() - bbSrc.br();
        int top = std::max(-tl.y, 0);  // Top and left are negated because adding a border
        int left = std::max(-tl.x, 0); // goes "the wrong way"
        int right = std::max(br.x, 0);
        int bottom = std::max(br.y, 0);
        cv::Mat tmp(src, validROI);
        cv::copyMakeBorder(tmp, dst, top, bottom, left, right, borderType);
    }
}


/*******************************************************************************************************************//**
* @brief This is so that we given an image and a region of interest and a color we draw a bounding box
* of that color around that region of interest in the image.
* This was an adaption from the pupil laps eye tracking code
* @author Krishna Bhattarai
***********************************************************************************************************************/
void PupilTracker::draw_dotted_rect(cv::Mat& image, const cv::Rect& rect , const cv::Scalar& color)
{
    int count = 0;
    auto create_Dotted_Line = [&](cv::Vec3b & pixel)
    {
        if (count % 4 == 0) {
            pixel[0] = color[0];
            pixel[1] = color[1];
            pixel[2] = color[2];
        }

        count++;
    };
    int x = rect.x;
    int y = rect.y;
    int width = rect.width - 1;
    int height = rect.height - 1;
    cv::Mat line  = image.colRange(x, width + 1).rowRange(y , y + 1);
    cv::Mat line2  = image.colRange(x, x + 1).rowRange(y , height + 1);
    cv::Mat line3  = image.colRange(x, width + 1).rowRange(height , height + 1);
    cv::Mat line4  = image.colRange(width, width + 1).rowRange(y , height + 1);
    std::for_each(line.begin<cv::Vec3b>(), line.end<cv::Vec3b>(), create_Dotted_Line);
    count = 0;
    std::for_each(line2.begin<cv::Vec3b>(), line2.end<cv::Vec3b>(), create_Dotted_Line);
    count = 0;
    std::for_each(line3.begin<cv::Vec3b>(), line3.end<cv::Vec3b>(), create_Dotted_Line);
    count = 0;
    std::for_each(line4.begin<cv::Vec3b>(), line4.end<cv::Vec3b>(), create_Dotted_Line);
}
