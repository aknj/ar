#ifndef _SHOW_TEST_HPP
#define _SHOW_TEST_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

inline void show_preview(cv::Mat & mat) {
    cv::imshow("preview", mat);
}

#endif
