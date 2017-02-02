#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

const int WIDTH = 320;
const int HEIGHT = 240;
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
    Mat frame, denoised, binary, canny, rgb_contours;
    namedWindow("input", 1);
    namedWindow("binary", 1);
    namedWindow("canny", 1);
    namedWindow("contours", 1);

    //- reading an image from file
    Mat img;

    img = imread("images/vip.jpg", CV_LOAD_IMAGE_COLOR);
    if(!img.data) {
        printf("Could not open or find the image");
        return -1;
    }

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
        cv::cvtColor(frame, frame, CV_BGR2GRAY);

        //-- smoothing (de-noising)
        // bilateralFilter(frame, blurred, 5, 100, 100);
        adaptiveBilateralFilter(frame, denoised, Size(5, 5), 100);
        // GaussianBlur(blurred, blurred, Size(3, 3), 0, 0);
        // medianBlur(frame, blurred, 5);

        adaptiveThreshold(blurred, blurred, 255, 
                          CV_ADAPTIVE_THRESH_MEAN_C, 
                          CV_THRESH_BINARY, 75, 10);
        
        // bitwise_not(blurred, blurred);

        cv::Canny(blurred, canny, 5, 20, 3);

        cvtColor(canny, rgb_contours, CV_GRAY2BGR);




        if(cv::waitKey(255) == 27)
            break;

        cap.retrieve(frame);

        

        
    }

    return 0;
}