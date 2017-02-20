#ifndef _HELPERS_HPP
#define _HELPERS_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;


float   perimeter(vector<cv::Point_<float>>&);

void    draw_polygon(cv::Mat, vector<cv::Point_<float>>&,
            cv::Scalar color = cv::Scalar(rand()%255, rand()%255, rand()%255));

cv::Mat bit_matrix_rotate(cv::Mat in);

int     marker_hamm_dist(const cv::Mat &bits);

int     matrix_to_id(const cv::Mat &bits);

#endif
