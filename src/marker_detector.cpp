#include "marker_detector.hpp"

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
