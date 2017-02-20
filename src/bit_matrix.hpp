#ifndef _BIT_MATRIX_HPP
#define _BIT_MATRIX_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/**************
 * bm_ = bit_matrix
 * in function names
 */

using namespace std;
using namespace cv;

int     bm_parity_check(const Mat & bits);

#endif