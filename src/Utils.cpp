#include "../include/Utils.h"
namespace fs = std::filesystem;


// Print coordinates of the tube 
void Utils::saveTubeSegmentCoords(vector<vector<ShapeCoordinates>>& tube_contour_coords, vector<float>& segment_length_incm) {

    // Create directory if it doesn't exist
    string folderPath = "../../tube-segment-coords";
    fs::create_directory(folderPath);

    // Loop through each segment (indexed from 0)
    for (size_t counter = 0; counter <= segment_length_incm.size(); ++counter) {

        // Generate file name and open file
        string fileName = "pointCloud" + to_string(counter) + ".asc";
        string filePath = folderPath + "/" + fileName;
        ofstream file(filePath);

        if (!file.is_open()) {
            cerr << "Failed to open file: " << filePath << endl;
            continue;
        }

        if (counter == 0) {
            for (const auto& point : tube_contour_coords[counter]) {
                file << 0 << " "
                     << (point.yVal * 1000) << " "
                     << (point.zVal * 1000) << "\n";
            }
            // Close the loop by writing the first point again
            const auto& first = tube_contour_coords[counter][0];
            file << 0 << " "
                 << (first.yVal * 1000) << " "
                 << (first.zVal * 1000) << "\n";
        } else {
            // Compute x coordinate
            float xVal = accumulate(segment_length_incm.begin(), segment_length_incm.begin() + counter, 0.0f) * 10;

            for (const auto& point : tube_contour_coords[counter - 1]) {
                file << (xVal) << " "
                     << (point.yVal * 1000) << " "
                     << (point.zVal * 1000) << "\n";
            }

            // Close the loop
            const auto& first = tube_contour_coords[counter - 1][0];
            file << (xVal) << " "
                 << (first.yVal * 1000) << " "
                 << (first.zVal * 1000) << "\n";
        }

        cout << "fileCounter = " << counter << "\n";
        file.close();
    }
}

// Implementing MATLAB linspace functionality
vector<float> Utils::linspace(float start, float end, int num_points) {
    
    vector<float> result;

    if (num_points == 0) return result;
    if (num_points == 1) {
        result.push_back(start);
        return result;
    }

    float step = (end - start) / (num_points - 1);

    for (int i = 0; i < num_points; ++i) {
        result.push_back(start + step * i);
    }

    return result;
}

// Find the intersection points of two shapes
vector<ShapeCoordinates> Utils::findShapeIntersections(const vector<ShapeCoordinates>& s1, const vector<ShapeCoordinates>& s2){

    // Build Boost linestrings from your vectors
    BoostLineString ls1, ls2;
    ls1.reserve(s1.size());
    ls2.reserve(s2.size());
    for (auto const& pt : s1) ls1.push_back( BoostPoint{pt.yVal, pt.zVal} );
    for (auto const& pt : s2) ls2.push_back( BoostPoint{pt.yVal, pt.zVal} );

    // Compute intersection
    BoostMultiPoint inter_pts;
    bg::intersection(ls1, ls2, inter_pts);

    // Convert back into ShapeCoordinates
    vector<ShapeCoordinates> result;
    result.reserve(inter_pts.size());
    for (auto const& p : inter_pts) {
        result.push_back({
          static_cast<float>( bg::get<0>(p) ),  // yVal
          static_cast<float>( bg::get<1>(p) )   // zVal
        });
    }
    return result;
}

// Compute area of a polygon
float Utils::computeTotalArea(const vector<ShapeCoordinates>& shape_coordinates) {

    BoostRing ring;
    ring.reserve(shape_coordinates.size() + 1);

    // Copy the contour into the ring
    for (auto const &pt : shape_coordinates) {
        ring.emplace_back(pt.yVal, pt.zVal);
    }

    // Make sure it’s closed contour (first == last)
    if (!bg::equals(ring.front(), ring.back())) {
        ring.push_back(ring.front());
    }

    // Compute area of the ring
    float halfArea = std::abs(bg::area(ring));
    return halfArea;
}