#include "join.hpp"

const map<int, int>
            marker_ids = { {106, 1},
                           {107, 2},
                           {108, 3},
                           {270, 4},
                           {300, 5},
                           {415, 6}
};

void place_images_and_show(const Mat&           frame,
                           vector<marker_t> &   markers,
                           vector<Mat>          imgs) {

    Mat markers_vis = frame.clone();

    for(size_t i = 0; i < markers.size(); i++) {
        marker_t& m = markers[i];
    
        if(marker_ids.find(m.id) == marker_ids.end()) {
            #ifdef DEBUG
            cout << "false marker id: " << m.id << endl << endl;
            #endif
            continue;
        }
    
        Mat img_warped = Mat::zeros(markers_vis.size(), markers_vis.type());

        warpPerspective(
            imgs[marker_ids.at(m.id)-1], img_warped, m.transform.inv(), 
            img_warped.size(), INTER_LINEAR,
            BORDER_REPLICATE, Scalar(100,100,100)
        );

        Mat mask = img_warped == 0;
        bitwise_and(mask, markers_vis, markers_vis);
        bitwise_or(img_warped, markers_vis, markers_vis);

        markers_vis = img_warped;

        // draw_polygon(markers_vis, m.points);
    }

    imshow("output", markers_vis);
}
