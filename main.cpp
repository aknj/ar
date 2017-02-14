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

// 320, 240
#ifdef STEPS
const int WIDTH = 320;
const int HEIGHT = 240;
#else
const int WIDTH = 320 * 2;
const int HEIGHT = 240 * 2;
#endif
const int FPS = 5;

const int marker_min_contour_length_allowed = 100;

const Size marker_size = Size(100,100);

//- marker corners for perspective transformation
static const Point2f arr[] = {
    Point2f(0,0),
    Point2f(marker_size.width-1,0),
    Point2f(marker_size.width-1,marker_size.height-1),
    Point2f(0,marker_size.height-1)
};
vector<Point2f> m_marker_corners2d(arr, arr + sizeof(arr) / sizeof(arr[0]));

typedef struct {
    vector<Point2f> points;
    int id;
    Mat transform;
} marker_t;

map<int, int> marker_ids = {
    {106, 1},
    {107, 2},
    {108, 3},
    {270, 4},
    {300, 5},
    {415, 6}
};


float perimeter(vector<Point2f> &a) {
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

/**
 * draw polygons with a random color of line
 */
void draw_polygon(Mat mat_name, vector<Point2f> &poly, 
                  Scalar color = Scalar(rand()%255, rand()%255, rand()%255)) 
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
            out.at<uchar>(i, j) = in.at<uchar>(in.cols-1-j, i);
        }
    }
    // cout << "in = " << endl << in << endl;
    // cout << "out = " << endl << out << endl;
    return out;
}

int marker_hamm_dist(Mat bits) {
    //- parity check matrix
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
    return dist;
}

int matrix_to_id(const Mat &bits) {
    int val = 0;
    for(int y = 0; y < 5; y++) {
        val <<= 1;
        if(bits.at<uchar>(y,2)) val|=1;
        val <<= 1;
        if(bits.at<uchar>(y,4)) val|=1;
    }
    return val ? val : -1;
}

int read_marker_id(Mat &marker_image, int &n_rotations) {
    assert(marker_image.rows == marker_image.cols);
    assert(marker_image.type() == CV_8UC1);

    Mat grey = marker_image;

    //- threshold image
    threshold(grey, grey, 125, 255, THRESH_BINARY | THRESH_OTSU);
    // namedWindow("binary marker", 1);

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

    //- check all possible rotations
    Mat bit_matrix_rotations[4];
    int distances[4];

    bit_matrix_rotations[0] = bit_matrix;
    distances[0] = marker_hamm_dist(bit_matrix_rotations[0]);

    pair<int,int> min_dist(distances[0], 0);

    for(int i = 1; i < 4; i++) {
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
        return matrix_to_id(bit_matrix_rotations[min_dist.second]);
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
    Mat frame, grayscale, thresholdImg, markers_prev, markers_vis, marker_image;
#ifdef STEPS
    namedWindow("input", 1);
    namedWindow("threshold", 1);
    namedWindow("contours_prev", 1);
    namedWindow("markers_cand", 1);
#endif
    namedWindow("output", 1);

    //- reading an image from file
    vector<Mat> imgs;

    for(int i = 0; i < 6; i++) {
        char path[20]; 
        sprintf(path, "images/%d.jpg", i+1);
        imgs.push_back(imread(path, CV_LOAD_IMAGE_COLOR));
        if(!imgs[i].data) {
            printf("Could not open or find the image %s", path);
            return -1;
        }
        resize(imgs[i], imgs[i], marker_size);
    }


    //- trackbars for changing the parameters of adaptiveThreshold
    int t1 = 111;
#ifdef STEPS
    createTrackbar("thr_blocksize", "contours_prev", &t1, 121);
#endif
    int t2 = 16;
#ifdef STEPS
    createTrackbar("thr_c", "contours_prev", &t2, 20);
#endif


    for(;;) {
        if(!cap.grab())
            continue;

        //- dismiss some frames
        it++;
        if(it % 10 != 0)
            continue;

        if(!cap.retrieve(frame) || frame.empty())
            continue;

        //- copy frame to marker visualization Mat
        markers_vis = frame.clone();

        //- manipulate frame
        cvtColor(frame, grayscale, CV_BGRA2GRAY);
        marker_image = grayscale.clone();

        
        int thr_blocksize = t1 / 2 * 2 + 3;
        int thr_c = t2 - 10;

        adaptiveThreshold(grayscale, thresholdImg, 255, 
                          CV_ADAPTIVE_THRESH_GAUSSIAN_C, 
                          CV_THRESH_BINARY_INV, thr_blocksize, thr_c);
        

        vector<vector<Point> > all_contours;
        vector<vector<Point> > contours;

        Mat contours_img;
        thresholdImg.copyTo(contours_img);
        findContours(contours_img, all_contours, CV_RETR_LIST, 
                     CV_CHAIN_APPROX_NONE);

        for(size_t i = 0; i < all_contours.size(); i++) {
            int contourSize = all_contours[i].size();
            if(contourSize > 4) {
                contours.push_back(all_contours[i]);
            }
        }

        Mat contours_prev = Mat::zeros(frame.size(), CV_8UC3);
        Mat markers_prev = Mat::zeros(frame.size(), CV_8UC3);

        drawContours(contours_prev, contours, -1, Scalar(255,0,0));

        //- find candidates -------
        vector<marker_t> possible_markers,
                         detected_markers,
                         good_markers;
        vector<Point> approx_curve;

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

            //- all tests are passed. save marker candidate
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

            if(o < 0.0)               //- if the 3rd point is on the left side,  
                swap(m.points[1], m.points[3]);        //--sort anti-clockwise 


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

        //-- mark the element of the pair with smaller perimeter for removal
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
            
            Mat canonical_marker_image = Mat(marker_size, grayscale.type());
            
            //- identify the markers
            for(size_t i=0; i < detected_markers.size(); i++) {
                marker_t marker;
                marker = detected_markers[i];

                //- find the perspective transformation that brings current
                //--marker to rectangular form
                Mat marker_transform = getPerspectiveTransform(
                                            marker.points, m_marker_corners2d
                );

                //- transform image to get a canonical marker image
                warpPerspective(grayscale, canonical_marker_image, 
                                marker_transform, marker_size);


                int n_rotations;
                int id = read_marker_id(canonical_marker_image, n_rotations);
                if(id != -1) {
                    marker.id = id;
                    marker.transform = marker_transform;
                    //- sort the points of the marker according to its data
                    std::rotate(marker.points.begin(),
                                marker.points.begin() + 4 - n_rotations,
                                marker.points.end() );

                    marker.transform = getPerspectiveTransform(
                        marker.points, m_marker_corners2d
                    );

                    good_markers.push_back(marker);
                }
            }
        }


        ////////////////////////////////////////////////////////////////////////
        //- refine marker corners using subpixel accuracy
        if(good_markers.size() > 0) {
            vector<Point2f> precise_corners(4 * good_markers.size());

            for(size_t i = 0; i < good_markers.size(); i++) {
                const marker_t& m = good_markers[i];

                for(int c = 0; c < 4; c++) {
                    precise_corners[i*4 + c] = m.points[c];
                }
            }

            TermCriteria term_criteria = TermCriteria(   
                TermCriteria::MAX_ITER | TermCriteria::EPS, 30, .01
            );
            cornerSubPix(   
                grayscale, precise_corners, Size(5,5), Size(-1,-1), 
                term_criteria
            );

            //-copy refined corners positions back to markers
            for(size_t i = 0; i < good_markers.size(); i++) {
                marker_t& m = good_markers[i];

                for(int c = 0; c < 4; c++) {
                    m.points[c] = precise_corners[i*4 + c];
                }
            }

            detected_markers = good_markers;
        
            ////////////////////////////////////////////////////////////////////
            //- operations on good markers
            for(size_t i = 0; i < detected_markers.size(); i++) {
                marker_t& m = detected_markers[i];
                
                ////////////////////////////////////////////////////////////////
                //- draw good markers' outlines and show their ids
                // char label[15];
                // sprintf(label, "#%lu, id=%d", i, m.id);
                // Scalar color = Scalar(rand()%255,rand()%255,rand()%255);

                // {
                //     draw_polygon(markers_vis, detected_markers[i].points, color);

                //     //- drawing numbers
                //     putText(markers_vis, label, m.points[1], 
                //             FONT_HERSHEY_SIMPLEX, .5, color);

                //     namedWindow("markers", 1);
                // }

                // if(m.id == 300) {
                //     putText(markers_vis, "wow!", m.points[3], 
                //             FONT_HERSHEY_SIMPLEX, .5, color);
                // }

                ////////////////////////////////////////////////////////////////
                //- place images on output frame
                Mat t = Mat::zeros(markers_vis.size(), markers_vis.type());
                
                // cout << m.id << "   " << marker_ids[m.id] << endl;
                warpPerspective(imgs[marker_ids[m.id]-1], t, m.transform.inv(), 
                                t.size());

                Mat mask = t == 0;
                bitwise_and(mask, markers_vis, markers_vis);
                bitwise_or(t, markers_vis, markers_vis);
            }
        }


        if(waitKey(255) == 27)
            break;

        // cap.retrieve(frame);


#ifdef STEPS
        imshow("input", frame);
        imshow("threshold", thresholdImg);
        imshow("contours_prev", contours_prev);
        imshow("markers_cand", markers_prev);
#endif
        imshow("output", markers_vis);

    }

    return 0;
}