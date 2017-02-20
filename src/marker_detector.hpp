#ifndef _MARKER_DETECTOR_HPP
#define _MARKER_DETECTOR_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


/******************************************************************************
    type definitions 
*/

typedef struct {
    vector<Point2f> points;
    int id;
    Mat transform;
} marker_t;


void prepare_image(const Mat &, Mat &);

void threshold(const Mat &, Mat &);

#endif
