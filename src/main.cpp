#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

#include "helpers.hpp"

using namespace std;
using namespace cv;

/******************************************************************************
    consts
*/

#ifdef STEPS
const int   WIDTH = 320;
const int   HEIGHT = 240;
#else
const int   WIDTH = 320 * 2;
const int   HEIGHT = 240 * 2;
#endif
const int   FPS = 60;

//- size and corners of the canonical marker
const Size  MARKER_SIZE = Size(100,100);
static
const Point2f
            PTS[] = { Point2f(0,0),
                      Point2f(MARKER_SIZE.width-1, 0),
                      Point2f(MARKER_SIZE.width-1, MARKER_SIZE.height-1),
                      Point2f(0, MARKER_SIZE.height-1)
};
const vector<Point2f>
            CANONICAL_M_CORNERS( PTS, PTS + sizeof(PTS)/sizeof(PTS[0]) );

const map<int, int>
            marker_ids = { {106, 1},
                           {107, 2},
                           {108, 3},
                           {270, 4},
                           {300, 5},
                           {415, 6}
};


/******************************************************************************
    type definitions 
*/

typedef struct {
    vector<Point2f> points;
    int id;
    Mat transform;
} marker_t;


/******************************************************************************
    functions
*/

void prepare_image(const Mat & bgra_mat, Mat & grayscale) {
    cvtColor(bgra_mat, grayscale, CV_BGRA2GRAY);
}

void threshold(const Mat & grayscale, Mat & threshold_img) {
    int t1 = 111;
    int t2 = 16;

    int thr_blocksize = t1 / 2 * 2 + 3;
    int thr_c = t2 - 10;

    adaptiveThreshold(grayscale, threshold_img, 255,
                        CV_ADAPTIVE_THRESH_GAUSSIAN_C,
                        CV_THRESH_BINARY_INV, thr_blocksize, thr_c);
}

void find_contours(const Mat & threshold_img, vector<vector<Point> > & contours,
                    int min_every_contour_length) {
    vector<vector<Point> > all_contours;

    Mat contours_img;
    threshold_img.copyTo(contours_img);
    findContours(threshold_img, all_contours, CV_RETR_LIST,
                    CV_CHAIN_APPROX_NONE);

    for(size_t i = 0; i < all_contours.size(); i++) {
        int contourSize = all_contours[i].size();
        if(contourSize > min_every_contour_length) {
            contours.push_back(all_contours[i]);
        }
    }
}

void find_possible_markers(const vector<vector<Point> >& contours,
                            vector<marker_t> & possible_markers) {
    vector<Point> approx_curve;
    const int MIN_M_CONTOUR_LENGTH_ALLOWED = 100;

    //-- for each contour, analyze if it is a parallelepiped likely to be 
    //---the marker
    for(size_t i = 0; i < contours.size(); i++) {
        //- approximate to a polygon
        double eps = contours[i].size() * 0.05;
        approxPolyDP(contours[i], approx_curve, eps, true);

        //- we're interested only in polygons that contain only 4 points
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
        if(min_dist < MIN_M_CONTOUR_LENGTH_ALLOWED)
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
    }
}

//- verify/recognize markers
void find_valid_markers(vector<marker_t> & detected_markers, 
                        vector<marker_t> & good_markers,
                        const Mat& grayscale) {
    
    Mat canonical_marker_image = Mat(MARKER_SIZE, grayscale.type());
    
    //- identify the markers
    for(size_t i=0; i < detected_markers.size(); i++) {
        marker_t& marker = detected_markers[i];

        //- find the perspective transformation that brings current
        //--marker to rectangular form
        Mat marker_transform = getPerspectiveTransform(
                                    marker.points, CANONICAL_M_CORNERS
        );

        //- transform image to get a canonical marker image
        warpPerspective(grayscale, canonical_marker_image,
                        marker_transform, MARKER_SIZE
        );


        int n_rotations;
        int id = read_marker_id(canonical_marker_image, n_rotations);
        if(id != -1) {
            marker.id = id;
            marker.transform = marker_transform;
            //- sort the points of the marker according to its data
            std::rotate(marker.points.begin(),
                        marker.points.begin() + 4 - n_rotations,
                        marker.points.end() 
            );

            marker.transform = getPerspectiveTransform(
                marker.points, CANONICAL_M_CORNERS
            );

            good_markers.push_back(marker);
        }
    }
}

//- refine marker corners using subpixel accuracy
void refine_using_subpix(vector<marker_t> & good_markers, const Mat& grayscale) {
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
}


int main() {
    VideoCapture cap(0);

    if(!cap.isOpened()) {
        printf("No camera detected\n");
        return 0;
    }

    //- set resolution & frame rate (FPS) --------------------------
    cap.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);
    cap.set(CV_CAP_PROP_FPS, FPS);

    int it = 0;
    Mat frame, grayscale, threshold_img, markers_vis,
        markers_prev, contours_prev;

    namedWindow("output", 1);

    //- reading images from files ----------------------------------
    vector<Mat> imgs;

    for(int i = 0; i < 6; i++) {
        char path[20];
        sprintf(path, "../images/%d.jpg", i+1);
        imgs.push_back(imread(path, CV_LOAD_IMAGE_COLOR));
        if(!imgs[i].data) {
            printf("Could not open or find the image %s", path);
            return -1;
        }
        resize(imgs[i], imgs[i], MARKER_SIZE);
    }

    //- main loop --------------------------------------------------
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
        prepare_image(frame, grayscale);
        threshold(grayscale, threshold_img);

        vector<vector<Point> > contours;

        //- populate the contours vector
        find_contours(threshold_img, contours, frame.cols / 5);

#ifdef STEPS
        contours_prev = Mat::zeros(grayscale.size(), CV_8UC3);
        drawContours(contours_prev, contours, -1, Scalar(255,0,0));
#endif

        //- find candidates -------
        vector<marker_t> possible_markers,
                         good_markers;

        find_possible_markers(contours, possible_markers);

#ifdef STEPS
        markers_prev = Mat::zeros(grayscale.size(), CV_8UC3);
        for(size_t i = 0; i < possible_markers.size(); i++) {
            draw_polygon(markers_prev, possible_markers[i].points);
        }
#endif

        find_valid_markers(possible_markers, good_markers, grayscale);

        if(good_markers.size() > 0) {
            refine_using_subpix(good_markers, grayscale);
        }

        //- for valid markers ---------------------------------
        for(size_t i = 0; i < good_markers.size(); i++) {
            marker_t& m = good_markers[i];

            if(marker_ids.find(m.id) == marker_ids.end()) {
                cout << "false marker id: " << m.id << endl << endl;
                continue;
            }

            //- place images on output frame ------------------
            Mat t = Mat::zeros(markers_vis.size(), markers_vis.type());

            warpPerspective( imgs[marker_ids.at(m.id)-1],
                                t,
                                m.transform.inv(),
                                t.size()
            );

            Mat mask = t == 0;
            bitwise_and(mask, markers_vis, markers_vis);
            bitwise_or(t, markers_vis, markers_vis);

            draw_polygon(markers_vis, m.points);
        }


        if(waitKey(155) == 27)
            break;


#ifdef STEPS
        imshow("input", frame);
        imshow("threshold", threshold_img);
        imshow("contours_prev", contours_prev);
        imshow("markers_cand", markers_prev);
#endif
        imshow("output", markers_vis);

    }

    return 0;
}
