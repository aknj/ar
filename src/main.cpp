#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

#include "marker_detector.hpp"
#include "fuse.hpp"

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

namedWindow("preview", 1);

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

    Mat frame;

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
        if(!cap.grab()) {
            continue; }

        if(!cap.retrieve(frame) || frame.empty()) {
            continue; }

        vector<marker_t> markers; 
        
        marker_detector(frame, markers);

        place_images_and_show(frame, markers, imgs);

        if(waitKey(55) == 27) {
            break; }
    }

    return 0;
}
