#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

const int WIDTH = 320 * 1.2;
const int HEIGHT = 240 * 1.2;
const int FPS = 5;

const int marker_min_contour_length_allowed = 100;

float perimeter(vector<cv::Point> &a)
{
  int dx, dy;
  float sum=0;
  
  for (size_t i=0;i<a.size();i++)
  {
    size_t i2=(i+1) % a.size();
    
    dx = a[i].x - a[i2].x;
    dy = a[i].y - a[i2].y;
    
    sum += sqrt(dx*dx + dy*dy);
  }
  
  return sum;
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

    int i = 0;
    Mat frame, grayscale, thresholdImg, markers_prev;
    namedWindow("input", 1);
    namedWindow("threshold", 1);
    namedWindow("contours_prev", 1);
    namedWindow("markers_prev", 1);

    //- reading an image from file
    Mat img;

    img = imread("images/dydelf.jpg", CV_LOAD_IMAGE_COLOR);
    if(!img.data) {
        printf("Could not open or find the image");
        return -1;
    }

    //- trackbars for changing the parameters of adaptiveThreshold
    int t1 = 111;
    createTrackbar("thr_blocksize", "contours_prev", &t1, 121);

    int t2 = 16;
    createTrackbar("thr_c", "contours_prev", &t2, 20);

    for(;;) {
        if(!cap.grab())
            continue;

        //- dismiss some frames
        i++;
        if(i % 30 != 0)
            continue;

        if(!cap.retrieve(frame) || frame.empty())
            continue;

        //- manipulate frame
        cvtColor(frame, grayscale, CV_BGRA2GRAY);

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
        vector<vector<Point> > detected_markers;
        vector<Point> approx_curve;
        vector<vector<Point> > possible_markers;

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
            vector<Point> m;

            for(int i = 0; i < 4; i++) 
                m.push_back(Point2f(approx_curve[i].x, approx_curve[i].y));

            //- sort the points in anti-clockwise order
            //- trace a line between the first and second point
            //- if the third point is at the right side, then the points are
            //--anti-clockwise
            Point v1 = m[1] - m[0];
            Point v2 = m[2] - m[0];

            double o = (v1.x * v2.y) - (v1.y * v2.x);

            if(o < 0.0)             //- if the 3rd point is in the left side, 
                swap(m[1], m[3]);   //--then sort in anti-clockwise order


            possible_markers.push_back(m);
            // printf("  x: %d, y: %d \n", m[1].x, m[1].y);

            Scalar color = Scalar(rand() % 255,rand() % 255,rand() % 255);
            line(markers_prev, m[0], m[1], color);
            line(markers_prev, m[1], m[2], color);
            line(markers_prev, m[2], m[3], color);
            line(markers_prev, m[3], m[0], color);
        }

        //-- remove these elements which corners are too close to each other
        //--- first detect candidate for removal:
        vector< pair<int,int> > too_near_candidates;
        for(size_t i = 0; i < possible_markers.size(); i++) {
            // vector<Point> m1; 
            // for(int c = 0; c < 4; c++)
            //     m1.push_back(possible_markers[i][c]);
            const vector<Point>& m1 = possible_markers[i];

            //- calculate the avg distance of each corner to the nearest corner
            //--of the other marker candidate
            for(size_t j = i+1; j < possible_markers.size(); j++) {
                // vector<Point> m2;
                // for(int c = 0; c < 4; c++)
                //     m2.push_back(possible_markers[j][c]);
                const vector<Point>& m2 = possible_markers[j];

                float dist_squared = 0;

                for(int c = 0; c < 4; c++) {
                    Point v = m1[c] - m2[c];
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
            float p1 = perimeter(possible_markers[too_near_candidates[i].first]);
            float p2 = perimeter(possible_markers[too_near_candidates[i].second]);

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
            if(!removal_mask[i])
                detected_markers.push_back(possible_markers[i]);
        }
    


        if(waitKey(255) == 27)
            break;

        cap.retrieve(frame);


        imshow("input", frame);
        imshow("threshold", thresholdImg);
        imshow("contours_prev", contours_prev);
        imshow("markers_prev", markers_prev);

    }

    return 0;
}