#ifndef _SHOW_TEST_HPP
#define _SHOW_TEST_HPP

#include "show_test.hpp"

namespace cv {
    inline void prepare_test_image(cv::Mat & mat) {
        cv::imshow("preview", mat);
        cv::waitKey(0);
    }
}

#endif
