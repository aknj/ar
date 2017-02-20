#include "bit_matrix.hpp"

int bm_parity_check(const Mat & bits) {
    unsigned char H_data[3][5] = {
        {1, 0, 1, 0, 1},           // the first parity check bit is inverted
        {0, 1, 1, 0, 0},
        {0, 0, 0, 1, 1}
    };
    Mat H = Mat(3, 5, CV_8UC1, H_data);

    Mat bits_c = bits.clone();
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