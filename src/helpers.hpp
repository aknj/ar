#ifndef _HELPERS_HPP
#define _HELPERS_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

#include "bit_matrix.hpp"

using namespace std;


float   perimeter(vector<cv::Point_<float>>&);

void    draw_polygon(cv::Mat, vector<cv::Point_<float>>&,
            cv::Scalar color = cv::Scalar(rand()%255, rand()%255, rand()%255));

void    show_preview(string name, const cv::Mat & mat);

int     read_marker_id(cv::Mat &marker_image, int &n_rotations);

#endif
