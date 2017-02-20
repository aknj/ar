#include "helpers.hpp"

using namespace cv;

/******************************************************************************
 ** calculates perimeter of a polygon *****************************************
 */
float perimeter(vector<cv::Point2f> &a) {
    float dx, dy;
    float sum = 0;
    
    for(size_t i = 0; i < a.size(); i++) {
        size_t i2 = (i+1) % a.size();
    
        dx = a[i].x - a[i2].x;
        dy = a[i].y - a[i2].y;
    
        sum += sqrt(dx*dx + dy*dy);
    }
  
    return sum;
}

/******************************************************************************
 ** draws polygons with a random color of line ********************************
 */
void draw_polygon(Mat mat_name, vector<Point2f> &poly, Scalar color)
{
    for(size_t i = 0; i < poly.size(); i++) {
        size_t i2 = (i+1) % poly.size();

        line(mat_name, poly[i], poly[i2], color);
    }
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
 ** compute hamming distance between bit matrix and all correct words
    TO DO: use hamming **********************
 */
int marker_hamm_dist(const Mat &bits) {
    //- all possible correct coded words
    bool words[4][5] = {
        {1, 0, 0, 0, 0},            // the first parity check bit is inverted
        {1, 1, 1, 1, 1},
        {0, 0, 0, 1, 1},
        {0, 1, 1, 0, 0}
    };

    int dist = 0;
    
    for (int y = 0; y < 5; y++) {
        int min_sum = 1e5;
        for(int p = 0; p < 4; p++) {
            int sum = 0;
            // counting
            for(int x = 0; x < 5; x++) {
                sum += bits.at<uchar>(y, x) == words[p][x] ? 0 : 1;
            }
            if(min_sum > sum)
                min_sum = sum;
        }
        dist += min_sum;
    }

    // for (int w = 0; w < 5; w++) {
    //     for(int x = 0; x < 5; x++) {
    //         bits.at<uchar>(w, x) = 0; // first parity bit
    //     }
    // }

    unsigned char H_data[3][5] = {
        {0, 0, 1, 0, 1},            // the first parity check bit is inverted
        {0, 1, 1, 0, 0},
        {0, 0, 0, 1, 1}
    };
    Mat H = Mat(3, 5, CV_8UC1, H_data);

    // unsigned char z[3];

    // vector<uchar> z;
    int error = 7;
    for(int j = 0; j < 5; j++) {
        vector<int> z;
        for(int i = 0; i < 3; i++) {
            vector<uchar> vec;
            bitwise_and(H.row(i), bits.row(j), vec);
            error = countNonZero(vec) % 2;  // if 0 then no error yet
            z.push_back(error);
        }
        cout << z[0] << " " << z[1] << " " << z[2] << endl;
    }

    //return countNonZero()

    // vector<uchar> z;
    // bitwise_and(H, bits.row(0), z);
    // cout << z[0] << z[1] << z[3] << endl;

    return dist;
}


// void marker_hamm_dist(const Mat &bits) {
//     //- all possible correct coded words
//     bool H[3][5] = {
//         {0, 0, 1, 0, 1},            // the first parity check bit is inverted
//         {0, 1, 1, 0, 0},
//         {0, 0, 0, 1, 1}
//     };

//     int dist = 0;
    

//     Mat z;
//     bitwise_and(H, bits, z);
//     return dist;
// }

/******************************************************************************
 **  ******************************************************
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
