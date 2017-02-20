#ifndef _BIT_MATRIX_HPP
#define _BIT_MATRIX_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/**************
 * bm_ = bit_matrix
 * in function names
 */

using namespace std;


int     bm_parity_check(const cv::Mat & bits);

cv::Mat bit_matrix_rotate(cv::Mat in);

int     matrix_to_id(const cv::Mat &bits);

#endif