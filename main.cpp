#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <bitset>
#include <algorithm>    // for copy
#include <iterator>     // for ostream_iterator

using namespace std;
using namespace cv;

const int WIDTH = 320 * 1.2;
const int HEIGHT = 240 * 1.2;
const int FPS = 5;

const int marker_min_contour_length_allowed = 100;
const Size marker_size = Size(100,100);
vector<Point2f> m_marker_corners2d;

typedef vector<Point2f> Marker;

typedef struct {
    vector<Point2f> points;
    int id;
} marker_t;



float perimeter(vector<Point2f> &a) {
    float dx, dy;
    float sum=0;
    
    for(size_t i=0; i < a.size(); i++) {
        size_t i2=(i+1) % a.size();
    
        dx = a[i].x - a[i2].x;
        dy = a[i].y - a[i2].y;
    
        sum += sqrt(dx*dx + dy*dy);
  }
  
  return sum;
}

/**
 * function draws polygons with a random color of line
 */
void draw_polygon(Mat mat_name, vector<Point2f> &poly, 
                  Scalar color = Scalar(rand() % 255,rand() % 255,rand() % 255)) 
{
    for(size_t i = 0; i < poly.size(); i++) {
        size_t i2 = (i+1) % poly.size();

        line(mat_name, poly[i], poly[i2], color);
    }
}

Mat bit_matrix_rotate(Mat in) {
    Mat out;
    in.copyTo(out);
    for(int i = 0; i < in.rows; i++) {
        for(int j = 0; j < in.cols; j++) {
            out.at<uchar>(i, j) = in.at<uchar>(j, in.rows-1-i);
        }
    }
    return out;
}

int marker_hamm_dist(Mat bits) {
    //- parity check matrix
    int H_data[3][5] = {
        {0, 0, 1, 0, 1},            // the first parity check bit is inverted
        {0, 1, 1, 0, 0},
        {0, 0, 0, 1, 1}
    };
    Mat H = Mat(3, 5, false, &H_data);
    
    int dist = 0;
    
    for(int p = 0; p < bits.rows; p++) {
        int min_sum = 1e5;
        vector<int> z;

        for(int i = 0; i < H.rows; i++) {
            int sum = 0;

            Mat bit_sum = H.row(i) & bits.row(p);
            
            z.push_back(countNonZero(bit_sum));

            // cout << "z = " << endl << " " << z << endl << endl;
        }

        copy(   z.begin(), 
                z.end(), 
                ostream_iterator<int>(cout, " ") );
        printf("\n");

    }

    return 0;
}

int matrix_to_id(const Mat &bits) {
    int val = 0;
    for(int y = 0; y < 5; y++) {
        val <<= 1;
        if(bits.at<uchar>(y,2)) val|=1;
        val <<= 1;
        if(bits.at<uchar>(y,4)) val|=1;
    }
    return val;
}

int read_marker_id(Mat &marker_image, int &n_rotations, int it) {
    assert(marker_image.rows == marker_image.cols);
    assert(marker_image.type() == CV_8UC1);

    Mat grey = marker_image;

    //- threshold image
    threshold(grey, grey, 125, 255, THRESH_BINARY | THRESH_OTSU);
    namedWindow("binary marker", 1);

    //- markers are divided in 7x7, of which the inner 5x5 belongs to marker
    //--info. the external border should be entirely black

    int cell_size = marker_image.rows / 7;

    for(int y=0; y < 7; y++) {
        int inc = 6;

        if(y==0 || y==6) inc = 1; // for 1st and last row, check whole border

        for(int x=0; x < 7; x+=inc) {
            int cellX = x * cell_size;
            int cellY = y * cell_size;
            Mat cell = grey(Rect(cellX, cellY, cell_size, cell_size));

            int n_z = countNonZero(cell);

            if(n_z > (cell_size * cell_size) / 2) {
                // return;
                return -1; // cannot be a marker bc the border elem is not black
            } 
        }
    }

    Mat bit_matrix = Mat::zeros(5, 5, CV_8UC1);

    //- get info (for each inner square, determiine if it is black or white)
    for(int y=0; y < 5; y++) {
        for(int x=0; x < 5; x++) {
            int cellX = (x+1) * cell_size;
            int cellY = (y+1) * cell_size;
            Mat cell = grey(Rect(cellX, cellY, cell_size, cell_size));

            int n_z = countNonZero(cell);
            if(n_z > (cell_size * cell_size) / 2)
                bit_matrix.at<uchar>(y, x) = 1;
        }
    }

    if(it%200 == 0) {
        for(int x = 0; x < 5; x++) {
            for(int y = 0; y < 5; y++) {
                printf(" %i", bit_matrix.at<uchar>(x, y));
            } 
            printf("\n");
        }
        printf("\n");
        imshow("binary marker", grey);
    }

    //- check all possible rotations
    Mat bit_matrix_rotations[4];
    int distances[4];

    bit_matrix_rotations[0] = bit_matrix;
    distances[0] = marker_hamm_dist(bit_matrix_rotations[0]);

    pair<int,int> min_dist(distances[0],0);

    for(int i=1; i < 4; i++) {
        //- get hamming distance
        bit_matrix_rotations[i] = bit_matrix_rotate(bit_matrix_rotations[i-1]);
        distances[i] = marker_hamm_dist(bit_matrix_rotations[i]);

        if(distances[i] < min_dist.first) {
            min_dist.first = distances[i];
            min_dist.second = i;
        }
    }

    n_rotations = min_dist.second;
    if(min_dist.first == 0) {
        printf("min_dist: %d", min_dist.first);
        return matrix_to_id(bit_matrix_rotations[min_dist.second]);
        // return 1;
    }

    return -1;
}


int main() {
    VideoCapture cap(0);

    if(!cap.isOpened()) {
        printf("No camera detected\n");
        return 0;
    }

    //- set resolution & frame rate (FPS)
    cap.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);
    cap.set(CV_CAP_PROP_FPS, FPS);

    int it = 0;
    Mat frame, grayscale, thresholdImg, markers_prev;
    namedWindow("input", 1);
    namedWindow("threshold", 1);
    namedWindow("contours_prev", 1);
    namedWindow("markers_cand", 1);

    //- reading an image from file
    Mat img;

    img = imread("images/zn1.png", CV_LOAD_IMAGE_COLOR);
    if(!img.data) {
        printf("Could not open or find the image");
        return -1;
    }

    //- trackbars for changing the parameters of adaptiveThreshold
    int t1 = 111;
    createTrackbar("thr_blocksize", "contours_prev", &t1, 121);

    int t2 = 16;
    createTrackbar("thr_c", "contours_prev", &t2, 20);


    m_marker_corners2d.push_back(Point2f(0,0));
    m_marker_corners2d.push_back(Point2f(marker_size.width-1,0));
    m_marker_corners2d.push_back(Point2f(marker_size.width-1,marker_size.height-1));
    m_marker_corners2d.push_back(Point2f(0,marker_size.height-1));

    for(;;) {
        if(!cap.grab())
            continue;

        //- dismiss some frames
        it++;
        if(it % 30 != 0)
            continue;

        if(!cap.retrieve(frame) || frame.empty())
            continue;

        //- manipulate frame
        cvtColor(frame, grayscale, CV_BGRA2GRAY);
        Mat marker_image = grayscale.clone();

        //-- smoothing (de-noising)
        // bilateralFilter(frame, denoised, 5, 100, 100);
        // adaptiveBilateralFilter(frame, denoised, Size(5, 5), 100);
        // GaussianBlur(blurred, blurred, Size(3, 3), 0, 0);
        // medianBlur(frame, blurred, 5);

        
        int thr_blocksize = t1 / 2 * 2 + 3;
        int thr_c = t2 - 10;

        adaptiveThreshold(grayscale, thresholdImg, 255, 
                          CV_ADAPTIVE_THRESH_GAUSSIAN_C, 
                          CV_THRESH_BINARY_INV, thr_blocksize, thr_c);
        

        vector<vector<Point> > allContours;
        vector<vector<Point> > contours;

        Mat contoursImg;
        thresholdImg.copyTo(contoursImg);
        findContours(contoursImg, allContours, CV_RETR_LIST, 
                     CV_CHAIN_APPROX_NONE);

        for(size_t i = 0; i < allContours.size(); i++) {
            int contourSize = allContours[i].size();
            if(contourSize > 4) {
                contours.push_back(allContours[i]);
            }
        }

        Mat contours_prev = Mat::zeros(frame.size(), CV_8UC3);
        Mat markers_prev = Mat::zeros(frame.size(), CV_8UC3);

        drawContours(contours_prev, contours, -1, Scalar(255,0,0));

        //- find candidates -------
        vector<marker_t> detected_markers;
        vector<Point> approx_curve;
        vector<marker_t> possible_markers;

        //-- for each contour, analyze if it is a parallelepiped likely to be 
        //---the marker
        for(size_t i = 0; i < contours.size(); i++) {
            //- approximate to a polygon
            double eps = contours[i].size() * 0.05;
            approxPolyDP(contours[i], approx_curve, eps, true);

            //- i'm interested only in polygons that contain only 4 points
            if(approx_curve.size() != 4)
                continue;

            //- and they have to be convex
            if(!isContourConvex(approx_curve))
                continue;

            //- ensure that the distance b/w consecutive points is large enough
            float min_dist = numeric_limits<float>::max();

            for(int i = 0; i < 4; i++) {
                Point side = approx_curve[i] - approx_curve[(i+1)%4];
                float squared_side_length = side.dot(side);
                min_dist = min(min_dist, squared_side_length);
            }

            //- check that distance is not very small
            if(min_dist < marker_min_contour_length_allowed)
                continue;

            //- all? tests are passed. save marker candidate
            marker_t m;

            for(int i = 0; i < 4; i++) 
                m.points.push_back(Point2f(approx_curve[i].x, approx_curve[i].y));

            //- sort the points in anti-clockwise order
            //- trace a line between the first and second point
            //- if the third point is at the right side, then the points are
            //--anti-clockwise
            Point v1 = m.points[1] - m.points[0];
            Point v2 = m.points[2] - m.points[0];

            double o = (v1.x * v2.y) - (v1.y * v2.x);

            if(o < 0.0)             //- if the 3rd point is in the left side, 
                swap(m.points[1], m.points[3]);   //--then sort in anti-clockwise order


            possible_markers.push_back(m);

            // draw_polygon(markers_prev, m);
        }

        //-- remove these elements which corners are too close to each other
        //--- first detect candidate for removal:
        vector< pair<int,int> > too_near_candidates;
        for(size_t i = 0; i < possible_markers.size(); i++) {
            const marker_t& m1 = possible_markers[i];

            //- calculate the avg distance of each corner to the nearest corner
            //--of the other marker candidate
            for(size_t j = i+1; j < possible_markers.size(); j++) {
                const marker_t& m2 = possible_markers[j];

                float dist_squared = 0;

                for(int c = 0; c < 4; c++) {
                    Point v = m1.points[c] - m2.points[c];
                    dist_squared += v.dot(v);
                }

                dist_squared /= 4;

                if(dist_squared < 100) {
                    too_near_candidates.push_back(pair<int,int>(i,j));
                }
            }
        }

        //-- mark for removal the element of the pair with smaller perimeter ???
        vector<bool> removal_mask(possible_markers.size(), false);

        for(size_t i = 0; i < too_near_candidates.size(); i++) {
            float p1 = perimeter(
                possible_markers[ too_near_candidates[i].first  ].points);
            float p2 = perimeter(
                possible_markers[ too_near_candidates[i].second ].points);

            size_t removal_index;
            if(p1 > p2)
                removal_index = too_near_candidates[i].second;
            else
                removal_index = too_near_candidates[i].first;

            removal_mask[removal_index] = true;
        }

        //-- return candidates
        detected_markers.clear();
        for(size_t i = 0; i < possible_markers.size(); i++) {
            if(!removal_mask[i]) {
                detected_markers.push_back(possible_markers[i]);
                draw_polygon(markers_prev, possible_markers[i].points);
            }
        }

        ////////////////////////////////////////////////////////////////////////
        //-- verify/recognize markers
        {
            vector<marker_t> good_markers;
            Mat canonical_marker_image;
            
            //- identify the markers
            for(size_t i=0; i < detected_markers.size(); i++) {
                marker_t marker;
                marker = detected_markers[i];

                //- find the perspective transformation that brings current
                //--marker to rectangular form
                

                // printf("%d", m_marker_corners2d.size());

                Mat marker_transform = getPerspectiveTransform(
                                            marker.points, m_marker_corners2d
                );

                //- transform image to get a canonical marker image
                warpPerspective(grayscale, canonical_marker_image, 
                                marker_transform, marker_size);

//# debug
                {
                    draw_polygon(marker_image, marker.points, Scalar(255, 0, 0));
                    Mat marker_sub_image = marker_image(boundingRect(marker.points));

                    namedWindow("markers", 1);
                }
//# enddebug

                // int n_rotations;
                // read_marker_id(canonical_marker_image, n_rotations, it);
                int n_rotations;
                int id = read_marker_id(canonical_marker_image, n_rotations, it);
                if(id != -1) {
                    marker.id = id;
                    //- sort the points of the marker according to its data
                    std::rotate(marker.points.begin(),
                                marker.points.begin() + 4 - n_rotations,
                                marker.points.end() );

                    good_markers.push_back(marker);
                }
            }

            detected_markers = good_markers;
        }
    
        //- overlay an image
        //- debug
        for(size_t i = 0; i < detected_markers.size(); i++) {
            char label[15];
            sprintf(label, "marker #%lu", i);
            printf(" %s\t", label);
            printf("id: %d\n", detected_markers[i].id);
        }

        if(waitKey(255) == 27)
            break;

        cap.retrieve(frame);


        imshow("input", frame);
        imshow("threshold", thresholdImg);
        imshow("contours_prev", contours_prev);
        imshow("markers_cand", markers_prev);
        imshow("markers", marker_image);

    }

    return 0;
}