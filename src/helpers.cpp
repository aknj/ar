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