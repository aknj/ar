#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

const int WIDTH = 320 * 1.2;
const int HEIGHT = 240 * 1.2;
const int FPS = 5;


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
    Mat frame, thresholdImg;
    namedWindow("input", 1);
    namedWindow("contours_prev", 1);

    //- reading an image from file
    Mat img;

    img = imread("images/vip.jpg", CV_LOAD_IMAGE_COLOR);
    if(!img.data) {
        printf("Could not open or find the image");
        return -1;
    }

    //- trackbars for changing the parameters of adaptiveThreshold
    int t1 = 75;
    createTrackbar("thr_blocksize", "contours_prev", &t1, 121);

    int t2 = 10;
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
        cvtColor(frame, frame, CV_BGR2GRAY);

        //-- smoothing (de-noising)
        // bilateralFilter(frame, denoised, 5, 100, 100);
        // adaptiveBilateralFilter(frame, denoised, Size(5, 5), 100);
        // GaussianBlur(blurred, blurred, Size(3, 3), 0, 0);
        // medianBlur(frame, blurred, 5);

        
        int thr_blocksize = t1 / 2 * 2 + 3;
        int thr_c = t2 - 10;

        adaptiveThreshold(frame, thresholdImg, 255, 
                          CV_ADAPTIVE_THRESH_GAUSSIAN_C, 
                          CV_THRESH_BINARY, thr_blocksize, thr_c);
        

        vector<vector<Point> > allContours;
        vector<vector<Point> > contours;
        findContours(thresholdImg, allContours, CV_RETR_LIST, 
                     CV_CHAIN_APPROX_NONE);

        for(size_t i = 0; i < allContours.size(); i++) {
            int contourSize = allContours[i].size();
            if(contourSize > 4) {
                contours.push_back(allContours[i]);
            }
        }

        Mat contours_prev = Mat::zeros(frame.size(), CV_8UC3);
        

        drawContours(contours_prev, contours, -1, Scalar(255,0,0));

        //- find candidates


        if(waitKey(255) == 27)
            break;

        cap.retrieve(frame);


        imshow("input", frame);
        imshow("contours_prev", contours_prev);


    }

    return 0;
}