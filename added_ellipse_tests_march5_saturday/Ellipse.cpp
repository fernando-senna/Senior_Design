
class Ellipse
{
public:
    // define ellipse variables
    cv::Point2f center; // (h, k)
    // each individual component of the center
    float center_x;     // h
    float center_y;     // k
    
    cv::Point2f radii;  // (major, minor)
    // each individual component of the radius
    float major_radius; // major or width
    float mionr_radius; // minor or height

    // ellipse orientation angle
    float angle;        // alpha
public:

    // constructors
    Ellipse();

    // accessors
    cv::RotatedRect getEllipseCircumference(const cv::RotatedRect& ellipse);
    cv::RotatedRect getEllipseRadii(const cv::RotatedRect& ellipse);
    cv::RotatedRect getEllipseAngle(const cv::RotatedRect& ellipse);
    cv::RotatedRect getEllipseCenter(const cv::RotatedRect& ellipse);


    cv::RotatedRect getEllipseRatio(const cv::RotatedRect& ellipse);
    cv::RotatedRect getEllipseMajorRadius(const cv::RotatedRect& ellipse);
    cv::RotatedRect getEllipseMinorRadius(const cv::RotatedRect& ellipse);
    cv::RotatedRect getEllipseCenterX(const cv::RotatedRect& ellipse);
    cv::RotatedRect getEllipseCenterY(const cv::RotatedRect& ellipse);

    // utility functions
    
};

    

#endif
