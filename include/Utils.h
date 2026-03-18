/*
 * Utils.h
 * 
 * Utility functions to generate the tube geometry.
 * 
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <numeric>
#include <filesystem>

#include "ShapeCoordinates.h"

#define BOOST_NO_EXCEPTIONS
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/ring.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/multi_point.hpp>
#include <boost/geometry/algorithms/intersection.hpp>

namespace bg = boost::geometry;

using BoostPoint      = bg::model::d2::point_xy<float>;
using BoostRing       = bg::model::ring<BoostPoint>;
using BoostLineString = bg::model::linestring<BoostPoint>;
using BoostMultiPoint = bg::model::multi_point<BoostPoint>;

using namespace std;

class Utils{
 
    public:
        void saveTubeSegmentCoords(vector<vector<ShapeCoordinates>>& tube_contour_coords, vector<float>& segment_length_incm);
        vector<float> linspace(float start, float end, int num_points);
        vector<ShapeCoordinates> findShapeIntersections(const vector<ShapeCoordinates>& s1, const vector<ShapeCoordinates>& s2);
        float computeTotalArea(const vector<ShapeCoordinates>& shape_coordinates);

};

#endif // UTILS_H_