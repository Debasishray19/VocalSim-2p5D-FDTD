/*
 * ShapeAlignment.h
 * 
 * Rotate the shape along the yz-plane and set the diameter of the shape along the y-axis
 * We define the diameter of a cross-section as the maximum distance between two points on the cross-section
 * 
 */

#ifndef SHAPEALIGNMENT_H_
#define SHAPEALIGNMENT_H_

#include <vector>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "ShapeCoordinates.h"

using namespace std;

class ShapeAlignment{

    public:
        size_t num_coordinates;
        vector<ShapeCoordinates> aligned_shape;
        float segment_diameter;

        ShapeCoordinates point1;
        ShapeCoordinates point2;

        thrust::device_vector<ShapeCoordinates> d_point1;
        thrust::device_vector<ShapeCoordinates> d_point2;
        thrust::device_vector<float> d_diameter;

        thrust::host_vector<ShapeCoordinates> h_point1;
        thrust::host_vector<ShapeCoordinates> h_point2;
        thrust::host_vector<float> h_diameter;

        thrust::device_vector<ShapeCoordinates> d_shape_coordinates;

    public:
        ShapeAlignment(vector<ShapeCoordinates>& shape_coordinates);

    public:
        void shapeDiameter();
        void rotateShapeAboutOrigin();

};

#endif // SHAPE_ALIGNMENT_H_