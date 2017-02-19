#ifndef _HELPERS_HPP
#define _HELPERS_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

float   perimeter(vector<cv::Point_<float>>&);

void    draw_polygon(cv::Mat, vector<cv::Point_<float>>&,
                Scalar color = Scalar(rand()%255, rand()%255, rand()%255));

Mat     bit_matrix_rotate(Mat in);

#endif
