#ifndef _JOIN_HPP
#define _JOIN_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "marker_detector.hpp"
#include "helpers.hpp"

using namespace cv;

void place_images_and_show(const Mat& f, vector<marker_t> &, vector<Mat> imgs);

#endif
