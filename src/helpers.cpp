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
void draw_polygon(Mat mat_name, vector<Point2f> &poly, Scalar color, int thickness)
{
    for(size_t i = 0; i < poly.size(); i++) {
        size_t i2 = (i+1) % poly.size();

        line(mat_name, poly[i], poly[i2], color, thickness);
    }
}


void show_preview(std::string name, const Mat & mat) {
    imshow(name, mat);
}


int read_marker_id(Mat &marker_image, int &n_rotations) {
    assert(marker_image.rows == marker_image.cols);
    assert(marker_image.type() == CV_8UC1);

    Mat grey = marker_image;

    //- threshold image
    threshold(grey, grey, 125, 255, THRESH_BINARY | THRESH_OTSU);

    //- markers are divided in 7x7, of which the inner 5x5 belongs to marker
    //--info. the external border should be entirely black
    int cell_size = marker_image.rows / 7;

    for(int y = 0; y < 7; y++) {
        int inc = 6;

        if(y==0 || y==6) inc = 1; // for 1st and last row, check whole border

        for(int x = 0; x < 7; x+=inc) {
            int cellX = x * cell_size;
            int cellY = y * cell_size;
            Mat cell = grey(Rect(cellX, cellY, cell_size, cell_size));

            int n_z = countNonZero(cell);

            if(n_z > (cell_size * cell_size) / 2) {
                return -1; // cannot be a marker bc the border elem is not black
            }
        }
    }

    Mat bit_matrix = Mat::zeros(5, 5, CV_8UC1);

    //- get info (for each inner square, determiine if it is black or white)
    for(int y = 0; y < 5; y++) {
        for(int x = 0; x < 5; x++) {
            int cellX = (x+1) * cell_size;
            int cellY = (y+1) * cell_size;
            Mat cell = grey(Rect(cellX, cellY, cell_size, cell_size));

            int n_z = countNonZero(cell);
            if(n_z > (cell_size * cell_size) / 2)
                bit_matrix.at<uchar>(y, x) = 1;
        }
    }

    //- check all possible rotations
    Mat bit_matrix_rotations[4];
    int distances[4];

    bit_matrix_rotations[0] = bit_matrix;
    distances[0] = bm_parity_check(bit_matrix_rotations[0]);

    pair<int,int> min_dist(distances[0], 0);

    for(int i = 1; i < 4; i++) {
        bit_matrix_rotations[i] = bit_matrix_rotate(bit_matrix_rotations[i-1]);
        distances[i] = bm_parity_check(bit_matrix_rotations[i]);

        if(distances[i] < min_dist.first) {
            min_dist.first = distances[i];
            min_dist.second = i;
        }
    }

    n_rotations = min_dist.second;
    if(min_dist.first == 0) {
        return matrix_to_id(bit_matrix_rotations[min_dist.second]);
    }

    return -1;
}
