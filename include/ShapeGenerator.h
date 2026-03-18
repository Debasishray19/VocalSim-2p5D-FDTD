/*
 * ShapeGenerator.h
 * 
 * Generates various shapes by intersecting regular shapes.
 * 
 */
#ifndef SHAPEGENERATOR_H_
#define SHAPEGENERATOR_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include "ShapeCoordinates.h"
#include "SimulationUnits.h"
#include "ShapeAlignment.h"
#include "Utils.h"

#define BOOST_NO_EXCEPTIONS
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/ring.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/multi_point.hpp>
#include <boost/geometry/algorithms/intersection.hpp>

class TubeGeometry;
using namespace std;

namespace bg = boost::geometry;

using BoostPoint      = bg::model::d2::point_xy<float>;
using BoostRing       = bg::model::ring<BoostPoint>;
using BoostLineString = bg::model::linestring<BoostPoint>;
using BoostMultiPoint = bg::model::multi_point<BoostPoint>;


class ShapeGenerator{

    private:
        vector<float>& s1_semi_major_axis_len;
        vector<float>& s1_semi_minor_axis_len;
        vector<float>& s2_semi_major_axis_len;
        vector<float>& s2_semi_minor_axis_len;
        vector<float>& s1_s2_center_distance;
        vector<float>& yIntersect;
        vector<float>& zIntersect;
        vector<ShapeCoordinates> s1_coordinates;
        vector<ShapeCoordinates> s2_coordinates;
        
    
    public:
        float segment_area;
        float segment_diameter;
        vector<ShapeCoordinates> full_shape_coordinates; // Generate coordinates to draw the full shape
        vector<ShapeCoordinates> aligned_full_shape_coordinates;

    public:

        ShapeGenerator(int num_tube_segments,
                       vector<float>& s1_semi_major_axis_len, 
                       vector<float>& s1_semi_minor_axis_len,
                       vector<float>& s2_semi_major_axis_len, 
                       vector<float>& s2_semi_minor_axis_len,
                       vector<float>& s1_s2_center_distance, 
                       vector<float>& yIntersect, 
                       vector<float>& zIntersect,
                       float segment_area);
        
        ShapeGenerator();

        ~ShapeGenerator();
    
    public:
        void combineRegularShapes(int shape_idx, int num_coordinates);
        void print_section_shape(vector<ShapeCoordinates> shape, string file_name);

    private:

        vector<float> linspace(float start, float end, int num_points);
        vector<ShapeCoordinates> generateEllipseCoordinates(float starting_coord_y, float ending_coord_y, 
                                                            float semi_major_axis_len, float semi_minor_axis_len, 
                                                            int num_points, float yCenter);

        vector<ShapeCoordinates> finalContourCoordinates(vector<ShapeCoordinates> s1_coordinates, vector<ShapeCoordinates> s2_coordinates, vector<ShapeCoordinates> intersection_points,
                                                         float s1_semi_major_axis_len, float s1_semi_minor_axis_len,
                                                         float s2_semi_major_axis_len, float s2_semi_minor_axis_len,
                                                         float yCenter1, float yCenter2, float s1_starting_coord_y, float s2_ending_coord_y, int num_half_coordinates);
};

#endif // SHAPEGENERATOR_H_