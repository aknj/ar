#ifndef _MARKER_DETECTOR_HPP
#define _MARKER_DETECTOR_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helpers.hpp"
#include "show_test.hpp"

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

/******************************************************************************
    consts
*/

//- size and corners of the canonical marker
extern const Size MARKER_SIZE;

/******************************************************************************
    function headers
*/

void marker_detector(const Mat & frame, vector<marker_t> &);

void prepare_image(const Mat &, Mat &);

void threshold(const Mat &, Mat &);

void find_contours(const Mat &, vector<vector<Point> > &, int);

void find_possible_markers(const vector<vector<Point> > &, vector<marker_t> &,
                            Mat preview_image);

void find_valid_markers(vector<marker_t> &, vector<marker_t> &, const Mat &);

void refine_using_subpix(vector<marker_t> &, const Mat &);


#endif
