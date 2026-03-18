/*
 * TubeGeometry.h
 * 
 * Construct the tube geometry
 * 
 */

#ifndef TUBEGEOMETRY_H_
#define TUBEGEOMETRY_H_

#include <cmath>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <math.h>
#include <algorithm>

#include "SimulationUnits.h"
#include "ShapeCoordinates.h"
#include "ShapeGenerator.h"
#include "Utils.h"

#define CIRCULAR 1
#define ELLIPTICAL 2
#define SINGLE_PLANE_SYMMETRIC 3

using namespace std;

// forward‐declare FDTDSolver 
class FDTDSolver;

class TubeGeometry{

    public:

        int tube_start_posX;
        int tube_start_posY;
        int tube_end_posX;
        int tube_end_posY;

        float mic_position;
        int mic_position_cells;

        int* upper_wall_pos;
        int* lower_wall_pos;

        vector<vector<ShapeCoordinates>> tube_shapes;
        vector<vector<ShapeCoordinates>> aligned_tube_shapes;

    private:

        int num_tube_segments;
        thrust::host_vector<int>h_tube_segment_diameter_incells;

        vector<float> tube_segment_length_incm;
        vector<float> cross_sectional_area_incm2;

        vector<float> tube_segment_length_inm;
        vector<float> tube_segment_diameter_inm;
        vector<float> major_axis_length_inm;
        vector<float> minor_axis_length_inm;
        vector<float> cross_sectional_area_inm2;

        float tube_length;
        vector<float> tube_segment_diameter; 

        int tube_length_incells;

        FDTDSolver& _fdtdSolver;

    public:

        // Import the tube geometry data using the constructor
        explicit TubeGeometry(FDTDSolver& fdtdSolver);

    public:
    
        void setComputationalDomain();
        void generateVocalTractWalls();
        void generateExcitationCells();
        void generateMouthEndCells();

        int getFrameH();
        int getFrameW();
        int getVocalTractShape();
        float getSpatialResolution();
        int getTubeLengthCells();
        int* getBoundarySegmentType();
        vector<float> getMinorAxisLength();
        vector<float> getTubeSegmentLengthInCM();
    
    private:
        
        vector<float> tubeAreaScaling(vector <float> tube_segment_diameter_inm);
        void computeTubeWidth();
        void computeTubeWidthCircle();
        void computeTubeWidthEllipse();
        void computeTubeWidthSinglePlaneSymmetry();
};

#endif // TUBEGEOMETRY_H_