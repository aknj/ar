#include "bit_matrix.hpp"

using namespace cv;

int bm_parity_check(const Mat & bits) {
    unsigned char H_data[3][5] = {
        {1, 0, 1, 0, 1},
        {0, 1, 1, 0, 0},
        {0, 0, 0, 1, 1}
    };
    Mat H = Mat(3, 5, CV_8UC1, H_data);

    Mat bits_c = bits.clone();      // the first parity check bit is inverted
    for(int i = 0; i < bits_c.rows; i++) {
        bits_c.at<uchar>(i, 0) = bits_c.at<uchar>(i, 0) == 0 ? 1 : 0;
    }


    int error = 0;
    for(int j = 0; j < 5 && error == 0; j++) {
        vector<int> z;
        for(int i = 0; i < 3; i++) {
            vector<uchar> vec;
            bitwise_and(H.row(i), bits_c.row(j), vec);
            error = countNonZero(vec) % 2;         // if 0 then no error yet
            z.push_back(error);
            if(error == 1)
                return 1;
        }
    }

    return 0;
}

/******************************************************************************
 ** rotates a bit matrix ******************************************************
 */
Mat bit_matrix_rotate(Mat in) {
    Mat out;
    in.copyTo(out);
    for(int i = 0; i < in.rows; i++) {
        for(int j = 0; j < in.cols; j++) {
            out.at<uchar>(i, j) = in.at<uchar>(in.cols-1-j, i);
        }
    }
    // cout << "in = " << endl << in << endl;
    // cout << "out = " << endl << out << endl;
    return out;
}

/******************************************************************************
 ** read id from bit matrix ***************************************************
 */
int matrix_to_id(const Mat &bits) {
    int val = 0;
    for(int y = 0; y < 5; y++) {
        val <<= 1;
        if(bits.at<uchar>(y, 2)) val|=1;
        val <<= 1;
        if(bits.at<uchar>(y, 4)) val|=1;
    }
    return val ? val : -1;
}