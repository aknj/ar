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

    #ifdef DEBUG
        if(marker_ids.find(m.id) == marker_ids.end()) {
            cout << "false marker id: " << m.id << endl << endl;
            continue;
        }
    #endif

        Mat t = Mat::zeros(markers_vis.size(), markers_vis.type());

        warpPerspective(
            imgs[marker_ids.at(m.id)-1], t, m.transform.inv(), t.size()
        );

        Mat mask = t == 0;
        bitwise_and(mask, markers_vis, markers_vis);
        bitwise_or(t, markers_vis, markers_vis);

        draw_polygon(markers_vis, m.points);
    }

    imshow("output", markers_vis);
}