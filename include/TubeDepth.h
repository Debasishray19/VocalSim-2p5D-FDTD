/*
 * TubeDepth.h
 * 
 * Derive tube depth across the sagittal plane.
 * 
 */

#ifndef TUBEDEPTH_H_
#define TUBEDEPTH_H_

#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <limits>

#include "ShapeCoordinates.h"
#include "ShapeGenerator.h"
#include "Utils.h"

using namespace std;

class TubeGeometry;

class TubeDepth{

    public:

        float min_depth;
        float open_space_depth;

    private:

        int frameH;
        int frameW;

        int tube_length_incells;
        int tube_start_posX;
        int tube_end_posX;

        int midY;
        float ds;

        TubeGeometry& _tubeGeometry;
        thrust::device_vector<float>& depthX;
        thrust::device_vector<float>& depthY;
        thrust::device_vector<float>& depthP;

    public:

        TubeDepth(TubeGeometry& tubeGeometry,
                  thrust::device_vector<float>& _depthX,
                  thrust::device_vector<float>& _depthY,
                  thrust::device_vector<float>& _depthP);

    public:
        
        int findMaxInColumnHost(const int* boundary_segment_type, int frameH, int frameW, int col);
        void setTubeDepth();
        void generateDepthP();
        void setMinDepth();
    
    private:

        void generateTubeDepthSinglePlaneSymmetricShape(vector<float>& temp_depthX, vector<float>& temp_depthY, vector<vector<ShapeCoordinates>>& tube_shapes,
                                                        int* boundary_segment_type, const int tube_length_incells,
                                                        const int* upper_wall_pos, const int* lower_wall_pos,
                                                        const int  tube_start_posX, const float ds, const int frameW, const int frameH);
        
};

#endif // TUBEDEPTH_H_