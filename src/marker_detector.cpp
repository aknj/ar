#include "marker_detector.hpp"


const Size MARKER_SIZE = Size(100,100);

const Point2f
            PTS[] = { Point2f(0,0),
                      Point2f(MARKER_SIZE.width-1, 0),
                      Point2f(MARKER_SIZE.width-1, MARKER_SIZE.height-1),
                      Point2f(0, MARKER_SIZE.height-1)
};
vector<Point2f>
            CANONICAL_M_CORNERS( PTS, PTS + sizeof(PTS)/sizeof(PTS[0]) );


void marker_detector(const Mat & frame, vector<marker_t> & markers) {
    Mat _gray, _thres;
    vector<vector<Point> > _contours;
    vector<marker_t> _possible_markers;

    prepare_image(frame, _gray);
    threshold(_gray, _thres);
    find_contours(_thres, _contours, frame.cols / 5);
    find_possible_markers(_contours, _possible_markers, frame.clone());
    find_valid_markers(_possible_markers, markers, _gray);
    if(markers.size() > 0) {
        refine_using_subpix(markers, _gray); }
}


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

void find_contours(const Mat & src, vector<vector<Point> > & contours,
                    int min_every_contour_length) {
    vector<vector<Point> > all_contours;

    Mat contours_img;
    src.copyTo(contours_img);
    findContours(contours_img, all_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    for(size_t i = 0; i < all_contours.size(); i++) {
        int contourSize = all_contours[i].size();
        if(contourSize > min_every_contour_length) {
            contours.push_back(all_contours[i]);
        }
    }

    drawContours(contours_img, contours, -1, Scalar(255,0,0));
    show_preview("contours preview", contours_img);
}

void find_possible_markers(const vector<vector<Point> > & contours,
                            vector<marker_t> & possible_markers,
                            Mat frame) {
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

    for(size_t i = 0; i < possible_markers.size(); i++) {
        draw_polygon(frame, possible_markers[i].points);
    }
    show_preview("possible markers preview", frame);
}

//- verify/recognize markers
void find_valid_markers(vector<marker_t> & detected_markers,
                        vector<marker_t> & good_markers,
                        const Mat & grayscale) {

    Mat canonical_marker_image = Mat(MARKER_SIZE, grayscale.type());
    Mat preview = grayscale.clone();

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

            draw_polygon(preview, marker.points);
            char label[10];
            sprintf(label, "id: %d", marker.id);
            putText(preview, label, marker.points[0], 
                    FONT_HERSHEY_SIMPLEX, .5, 
                    Scalar(rand()%255, rand()%255, rand()%255)
            );
        }   
    }
    show_preview("markers preview", preview);
}

//- refine marker corners using subpixel accuracy
void refine_using_subpix(vector<marker_t> & good_markers, const Mat & grayscale) {
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

