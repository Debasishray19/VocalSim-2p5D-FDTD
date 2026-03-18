#include "../include/TubeGeometry.h"
#include "../include/TubeDepth.h"


// Check error codes for CUDA functions
#define CHECK(call){                                           \
    cudaError_t error = call;                                  \
    if (error != cudaSuccess){                                 \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error,       \
        cudaGetErrorString(error));                            \
    }                                                          \
}

__global__
void generateTubeDepthCircularShape(float* depthX, float* depthY,
const int* upper_wall_pos, const int* lower_wall_pos,
const int tube_length_incells,
const int tube_start_posX, const float ds, int frameW);

__global__
void generateTubeDepthEllipticalShape(float* depthX, float* depthY, float* minor_axis_length,
int* boundary_segment_type,
const int* upper_wall_pos, const int* lower_wall_pos,
const int tube_length_incells,
const int tube_start_posX, const float ds, 
int frameW, int frameH);

__device__
float safe_sqrtf(float sqr_val,
int tid,
int posY,
int posX,
const char* var_name);

__device__ inline int idx2d(int row, int col, int frameW);

__global__
void resetOpenSpaceDepth(float* depthX, float* depthY, int midY, int tube_end_posX, int frameH, int frameW, const float open_space_depth);

__global__
void smoothDepthX(
const float* depthX,    
float* temp_depthX,   
int frameH,
int frameW,
int midY,
int tubeEndX
);

__global__
void smoothDepthY(
const float* depthY,    
float* temp_depthY,   
int frameH,
int frameW,
int midY,
int tubeEndX
);

__global__
void updateDepthP(float* depthP, float* depthX, float* depthY,
                  int frameH, int frameW, int midY, int tubeEndX);

__global__
void setMinDepthVals(float* depthX, float* depthY, float* depthP, float min_depth, int frameH, int frameW);                  

void print_section_shape(vector<ShapeCoordinates> shape, string file_name){

    string file_path = "../../results/" + file_name;

    ofstream out(file_path);
    if (!out) {
        std::cerr << "Error: could not open the file for writing\n";
    } else {
        // column header (optional)
        out << "# yVal    zVal\n";
        for (const auto &pt : shape) {
          out << pt.yVal  << ' '<< pt.zVal << '\n';
        }  
    }
    out.close();
}

TubeDepth::TubeDepth(TubeGeometry& tubeGeometry, 
                     thrust::device_vector<float>& _depthX, 
                     thrust::device_vector<float>& _depthY,
                     thrust::device_vector<float>& _depthP):_tubeGeometry(tubeGeometry), depthX(_depthX), depthY(_depthY), depthP(_depthP)
{
    min_depth = 0.001f;
    open_space_depth = 0.50f;
}


void TubeDepth::setTubeDepth(){

    frameH = _tubeGeometry.getFrameH();
    frameW = _tubeGeometry.getFrameW();

    tube_length_incells = _tubeGeometry.getTubeLengthCells();
    tube_start_posX = _tubeGeometry.tube_start_posX;
    tube_end_posX = _tubeGeometry.tube_end_posX;

    midY = _tubeGeometry.tube_start_posY;
    ds = _tubeGeometry.getSpatialResolution();

    // Set the default tube depth values
    thrust::fill(depthP.begin(), depthP.end(), 1.0*open_space_depth);
    thrust::fill(depthX.begin(), depthX.end(), 1.0*open_space_depth);
    thrust::fill(depthY.begin(), depthY.end(), 1.0*open_space_depth);

    //Set thread configuration
    int NUM_THREADS = tube_length_incells;
    int NUM_THREADS_PER_BLOCK = 16;
    int NUM_BLOCKS = (NUM_THREADS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

    if(_tubeGeometry.getVocalTractShape()==CIRCULAR){
        
        // Compute depthX and depthY
        generateTubeDepthCircularShape<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(depthX.data()),
                                                                              thrust::raw_pointer_cast(depthY.data()),
                                                                              _tubeGeometry.upper_wall_pos, 
                                                                              _tubeGeometry.lower_wall_pos,
                                                                              tube_length_incells,
                                                                              tube_start_posX, ds, frameW);
                                                                            
        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess){
            fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(cerr));
            exit(EXIT_FAILURE);
        }

        // Generate depthP using depthX and depthY
        generateDepthP();

        // Set min_depth for depthX, depthY and depthP
        setMinDepth();


    }else if(_tubeGeometry.getVocalTractShape()==ELLIPTICAL){
        
        // Get the boundary segment types
        int* boundary_segment_type = nullptr;
        boundary_segment_type = _tubeGeometry.getBoundarySegmentType();

        // Transfer ellipse minor axis length to device
        thrust::device_vector<float> minor_axis_length(_tubeGeometry.getMinorAxisLength().begin(), _tubeGeometry.getMinorAxisLength().end());
        
        // Compute depthX and depthY
        generateTubeDepthEllipticalShape<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(depthX.data()),
                                                                                thrust::raw_pointer_cast(depthY.data()),
                                                                                thrust::raw_pointer_cast(minor_axis_length.data()),
                                                                                boundary_segment_type,
                                                                                _tubeGeometry.upper_wall_pos, 
                                                                                _tubeGeometry.lower_wall_pos,
                                                                                tube_length_incells,
                                                                                tube_start_posX, ds, frameW, frameH);
        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess){
            fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(cerr));
            exit(EXIT_FAILURE);
        }

        // Generate depthP using depthX and depthY
        generateDepthP();

        // Set min_depth for depthX, depthY and depthP
        setMinDepth();

    }else if(_tubeGeometry.getVocalTractShape()==SINGLE_PLANE_SYMMETRIC){

        // For simplicity, I don't not use a CUDA kernel to compute depth for this case
        // The SINGLE_PLANE_SYMMETRIC case is different from CIRCULAR and ELLIPTICAL.
        // Here, I use the complete cross-sectional shape [i.e., coordinates of a cross-section]
        // to compute the depthX and depthY values.

        // Get the boundary segment types
        int* d_boundary_segment_type = nullptr;
        d_boundary_segment_type = _tubeGeometry.getBoundarySegmentType();

        // Copy the boundary_segment_type from device to host memory
        size_t num_interior_cells = frameH * frameW;
        int* h_boundary_segment_type = new int[num_interior_cells];
        cudaMemcpy(h_boundary_segment_type, d_boundary_segment_type, num_interior_cells * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Copy upper_wall_pos and lower_wall_pos yo host memory
        num_interior_cells = _tubeGeometry.getTubeLengthCells();
        int* h_upper_wall_pos = new int[num_interior_cells];
        int* h_lower_wall_pos = new int[num_interior_cells];

        cudaMemcpy(h_upper_wall_pos, _tubeGeometry.upper_wall_pos, num_interior_cells * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_lower_wall_pos, _tubeGeometry.lower_wall_pos, num_interior_cells * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Compute depthX and depthY
        vector<float> temp_depthX(frameH*frameW, 1.0f * open_space_depth);
        vector<float> temp_depthY(frameH*frameW, 1.0f * open_space_depth);
        
        generateTubeDepthSinglePlaneSymmetricShape(temp_depthX, temp_depthY, _tubeGeometry.aligned_tube_shapes,
                                                   h_boundary_segment_type, tube_length_incells,
                                                   h_upper_wall_pos, h_lower_wall_pos,
                                                   tube_start_posX, ds, frameW, frameH);
        
        // Copy the temp_depthX and temp_depthY vectors to depthX and depthY device vectors
        depthX = temp_depthX;
        depthY = temp_depthY;
        
        // Generate depthP using depthX and depthY
        generateDepthP();

        // Set min_depth for depthX, depthY and depthP
        setMinDepth();
    }
}

void TubeDepth::generateDepthP(){
    
    // Reset grid cells with open_space_depth to the depthX[midY, tube_end_posX]
    int NUM_THREADS = frameH * frameW;
    int NUM_THREADS_PER_BLOCK = 256;
    int NUM_BLOCKS = (NUM_THREADS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    
    resetOpenSpaceDepth<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(depthX.data()),
                                                               thrust::raw_pointer_cast(depthY.data()),
                                                               midY, tube_end_posX, frameH, frameW, open_space_depth);
    CHECK(cudaGetLastError());

    // Smooth out depthX and depthY values between adjacent cells
    thrust::device_vector<float> temp_depthX(frameH*frameW), temp_depthY(frameH*frameW);

    smoothDepthX<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(depthX.data()),
                                                         thrust::raw_pointer_cast(temp_depthX.data()),
                                                         frameH, frameW, midY, tube_end_posX);
    CHECK(cudaGetLastError());

    smoothDepthY<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(depthY.data()),
                                                         thrust::raw_pointer_cast(temp_depthY.data()),
                                                         frameH, frameW, midY, tube_end_posX);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Swap the depthX
    thrust::swap(depthX, temp_depthX);
    thrust::swap(depthY, temp_depthY);
    
    // Compute depthP
    updateDepthP<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(depthP.data()),
                                                        thrust::raw_pointer_cast(depthX.data()),
                                                        thrust::raw_pointer_cast(depthY.data()),
                                                        frameH, frameW, midY, tube_end_posX);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

}

void TubeDepth::setMinDepth(){

    int NUM_THREADS = frameH * frameW;
    int NUM_THREADS_PER_BLOCK = 256;
    int NUM_BLOCKS = (NUM_THREADS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

    setMinDepthVals<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(depthX.data()),
                                                           thrust::raw_pointer_cast(depthY.data()),
                                                           thrust::raw_pointer_cast(depthP.data()),
                                                           min_depth, frameH, frameW);
}

// Shorthand for indexing into row/col of a flattened array of size frameW×frameH
__device__ inline int idx2d(int row, int col, int frameW) {
    return row*frameW + col;
}

// Helper function to compute square root safely
__device__
float safe_sqrtf(float sqr_val,
                 int tid,
                 int posY,
                 int posX,
                 const char* var_name)
{
    if (sqr_val < 0.f) {
        // Print out enough context to debug
        printf("FATAL [%s]: thread %d, posY=%d, posX=%d, got sqr_val=%f\n",
               var_name, tid, posY, posX, sqr_val);
        // immediately kill the kernel
        asm("trap;");
    }
    return static_cast<float>(sqrt(sqr_val));
}

// Device function to find Max in column
__device__ __inline__
int findMaxInColumnDevice(const int* boundary_segment_type, int frameH, int frameW, int col) {
    int m = 0;
    for(int y = 0; y < frameH; ++y) {
        int idx = y*frameW + col;
        m = max(m, boundary_segment_type[idx]);
    }
    return m;
}

// Host function to find Max in column
int TubeDepth::findMaxInColumnHost(const int* boundary_segment_type, int frameH, int frameW, int col){
    int m = 0;
    for (int y = 0; y < frameH; ++y) {
        int idx = y * frameW + col;
        m = std::max(m, boundary_segment_type[idx]);   
    }
    return m;
}


// CUDA Kernel to derive depthX and depthY
__global__
void generateTubeDepthCircularShape(float* depthX, float* depthY,
                                    const int* upper_wall_pos, const int* lower_wall_pos,
                                    const int tube_length_incells,
                                    const int tube_start_posX, const float ds, int frameW){
    
    // Global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= tube_length_incells) return;

    // Find the Y-coordinates of the upper and lower walls
    int upper_wallY = upper_wall_pos[tid];
    int lower_wallY = lower_wall_pos[tid];

    // Find the X-coordinate of the wall_pos
    int posX = tube_start_posX + tid;

    // Number of air cells between the upper and lower walls
    int air_cells_count = lower_wallY - upper_wallY - 1;
    int half_air_cells_count = (int)ceilf( air_cells_count*0.5f ) - 1;

    // Get the radius
    float radius = (static_cast<float>(air_cells_count)/2.0f) * ds;  
    float sqr_radius = radius * radius; 

    int cell_counter = half_air_cells_count;

    while(cell_counter >= 0){

        // Find Y-coordinates for depthX and depthY
        int posUpperY_depthX = upper_wallY + (half_air_cells_count - cell_counter) + 1;
        int posLowerY_depthX = lower_wallY - (half_air_cells_count - cell_counter) - 1;
        int posUpperY_depthY = upper_wallY + (half_air_cells_count - cell_counter) + 1;
        int posLowerY_depthY = lower_wallY - (half_air_cells_count - cell_counter) - 1;

        int posUpperY_depthX_idx = idx2d(posUpperY_depthX, posX, frameW);
        int posLowerY_depthX_idx = idx2d(posLowerY_depthX, posX, frameW);
        int posUpperY_depthY_idx = idx2d(posUpperY_depthY, posX, frameW);
        int posLowerY_depthY_idx = idx2d(posLowerY_depthY, posX, frameW);

        // Find depthX
        float distance = static_cast<float>(cell_counter) * ds;
        float sqr_distanceX = distance * distance;
        float sqr_depthX = sqr_radius - sqr_distanceX;
        float depth_X = safe_sqrtf(sqr_depthX, tid, posUpperY_depthX, posX, "sqr_depthX");

        // Find depthY
        float distance_lowerY = (static_cast<float>(cell_counter) - 0.5f) * ds;
        float distance_upperY = (static_cast<float>(cell_counter) + 0.5f) * ds;
        float sqr_distance_lowerY = distance_lowerY * distance_lowerY;
        float sqr_distance_upperY = distance_upperY * distance_upperY;

        float sqr_depth_lowerY = sqr_radius - sqr_distance_lowerY;
        float sqr_depth_upperY = 0.0f;

        if(cell_counter != half_air_cells_count)
            sqr_depth_upperY = sqr_radius - sqr_distance_upperY;

        float depth_lowerY = safe_sqrtf(sqr_depth_lowerY, tid, posLowerY_depthY, posX, "sqr_depth_lowerY");
        float depth_upperY = safe_sqrtf(sqr_depth_upperY, tid, posUpperY_depthY, posX, "sqr_depth_upperY");

        depthX[posUpperY_depthX_idx] = 2 * depth_X;
        depthX[posLowerY_depthX_idx] = 2 * depth_X;
        depthY[posUpperY_depthY_idx] = 2 * depth_upperY;
        depthY[posLowerY_depthY_idx] = 2 * depth_lowerY;

        cell_counter = cell_counter - 1;
    }

    // Assign tube depth to zero for boundary cells
    int lowerY_idx = idx2d(lower_wallY, posX, frameW);
    int upperY_idx = idx2d(upper_wallY, posX, frameW);

    depthX[upperY_idx] = 0;
    depthX[lowerY_idx] = 0;
    depthY[upperY_idx] = 0;
    depthY[lowerY_idx] = 0;
}
__global__
void generateTubeDepthEllipticalShape(float* depthX, float* depthY, float* minor_axis_length,
                                      int* boundary_segment_type,
                                      const int* upper_wall_pos, const int* lower_wall_pos,
                                      const int tube_length_incells,
                                      const int tube_start_posX, const float ds, 
                                      int frameW, int frameH){

    // Global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= tube_length_incells) return;

    // Find the Y-coordinates of the upper and lower walls
    int upper_wallY = upper_wall_pos[tid];
    int lower_wallY = lower_wall_pos[tid];

    // Find the X-coordinate of the wall_pos
    int posX = tube_start_posX + tid;  
    
    // Number of air cells between the upper and lower walls
    int air_cells_count = lower_wallY - upper_wallY - 1;
    int half_air_cells_count = (int)ceilf( air_cells_count*0.5f ) - 1;
    
    // Find the major and minor axis length
    int max_boundary_seg_type = findMaxInColumnDevice(boundary_segment_type, frameH, frameW, posX);

    float semi_major_axis_len = (air_cells_count*ds)/2.0f;
    float semi_minor_axis_len=  minor_axis_length[max_boundary_seg_type-1]/2.0f; 

    float sqr_semi_major_axis_len = semi_major_axis_len * semi_major_axis_len;
    float sqr_semi_minor_axis_len = semi_minor_axis_len * semi_minor_axis_len;

    int cell_counter = half_air_cells_count;

    while(cell_counter >= 0){

        // Find Y-coordinates for depthX and depthY
        int posUpperY_depthX = upper_wallY + (half_air_cells_count - cell_counter) + 1;
        int posLowerY_depthX = lower_wallY - (half_air_cells_count - cell_counter) - 1;
        int posUpperY_depthY = upper_wallY + (half_air_cells_count - cell_counter) + 1;
        int posLowerY_depthY = lower_wallY - (half_air_cells_count - cell_counter) - 1;

        int posUpperY_depthX_idx = idx2d(posUpperY_depthX, posX, frameW);
        int posLowerY_depthX_idx = idx2d(posLowerY_depthX, posX, frameW);
        int posUpperY_depthY_idx = idx2d(posUpperY_depthY, posX, frameW);
        int posLowerY_depthY_idx = idx2d(posLowerY_depthY, posX, frameW);

        // To calculate depth we will use the ellipse equation i.e.,
        // y^2/a^2 + depth^2/b^2 = 1
        // [Note]: depth = height along the z axis
        // depth = b*sqrt(1 - y^2/a^2)
            
        // For depthX,
        // y^2 = sqrDistanceX
        // a^2 = sqrSemiMajorAxis = semiMajorAxisLength^2
        // b = semiMinorAxisLength

        // Find depthX
        float distance = static_cast<float>(cell_counter) * ds;
        float sqr_distanceX = distance * distance;
        float sqr_depthX = sqr_semi_minor_axis_len * (1.0f - (sqr_distanceX/sqr_semi_major_axis_len));
        float depth_X = safe_sqrtf(sqr_depthX, tid, posUpperY_depthX, posX, "sqr_depthX");

        //printf("sqr_semi_minor_axis_len = %0.5f sqr_semi_major_axis_len = %0.5f\n", sqr_semi_minor_axis_len, sqr_semi_major_axis_len);
        //printf("distance = %0.5f sqr_distanceX = %0.5f sqr_depthX = %0.5f depth_X = %0.5f\n", distance, sqr_distanceX, sqr_depthX, depth_X);

        // Find depthY
        float distance_lowerY = (static_cast<float>(cell_counter) - 0.5f) * ds;
        float distance_upperY = (static_cast<float>(cell_counter) + 0.5f) * ds;
        float sqr_distance_lowerY = distance_lowerY * distance_lowerY;
        float sqr_distance_upperY = distance_upperY * distance_upperY;

        float sqr_depth_lowerY = sqr_semi_minor_axis_len*(1 - (sqr_distance_lowerY/sqr_semi_major_axis_len));
        float sqr_depth_upperY = 0.0f;

        if(cell_counter != half_air_cells_count)
            sqr_depth_upperY = sqr_semi_minor_axis_len*(1 - (sqr_distance_upperY/sqr_semi_major_axis_len));
        
        float depth_lowerY = safe_sqrtf(sqr_depth_lowerY, tid, posLowerY_depthY, posX, "sqr_depth_lowerY");
        float depth_upperY = safe_sqrtf(sqr_depth_upperY, tid, posUpperY_depthY, posX, "sqr_depth_upperY");
    
        depthX[posUpperY_depthX_idx] = 2 * depth_X;
        depthX[posLowerY_depthX_idx] = 2 * depth_X;
        depthY[posUpperY_depthY_idx] = 2 * depth_upperY;
        depthY[posLowerY_depthY_idx] = 2 * depth_lowerY;

        cell_counter = cell_counter - 1;
    }

    // Assign tube depth to zero for boundary cells
    int lowerY_idx = idx2d(lower_wallY, posX, frameW);
    int upperY_idx = idx2d(upper_wallY, posX, frameW);

    depthX[upperY_idx] = 0;
    depthX[lowerY_idx] = 0;
    depthY[upperY_idx] = 0;
    depthY[lowerY_idx] = 0;
}

// Derive depthX and depthY for single plane symmetric shape
void TubeDepth::generateTubeDepthSinglePlaneSymmetricShape(vector<float>& temp_depthX, vector<float>& temp_depthY, vector<vector<ShapeCoordinates>>& aligned_tube_shapes,
                                                           int* boundary_segment_type, const int tube_length_incells,
                                                           const int* upper_wall_pos, const int* lower_wall_pos,
                                                           const int  tube_start_posX, const float ds, const int frameW, const int frameH){
    
    // Create two ShapeCoordinates vector to store the cross-section and line coordinates
    vector<ShapeCoordinates> curr_vt_shape;
    vector<ShapeCoordinates> line_z;
    
    // Create an instance of Utility class to access its functions
    Utils utils;

    float depthX_val = 0;
    float depthY_val = 0;
    
    // For each cross-section, traverse along the y axis of the vocal tract to compute depthX and depthY
    // At the desired location, draw a line perpendicular to the y axis
    // to find the intersection points along the z-axis
                                                          
    for(int tube_length_idx=0; tube_length_idx < tube_length_incells; tube_length_idx++){

        // Find the Y-coordinates of the upper and lower walls
        int upper_wallY = upper_wall_pos[tube_length_idx];
        int lower_wallY = lower_wall_pos[tube_length_idx];

        // Find the X-coordinate of the wall_pos
        int posX = tube_start_posX + tube_length_idx;

        // Number of air cells between the upper and lower walls
        int air_cells_count = lower_wallY - upper_wallY - 1;

        // Find the major and minor axis length
        int max_boundary_seg_type = findMaxInColumnHost(boundary_segment_type, frameH, frameW, posX);

        // Extract the vocal tract cross-section
        curr_vt_shape = aligned_tube_shapes[max_boundary_seg_type-1];

        // Find the min and max zVal of the cross-section
        float minZ =  numeric_limits<float>::infinity();
        float maxZ = -numeric_limits<float>::infinity();

        // Find minY to traverse through the y-axis
        float minY =  numeric_limits<float>::infinity();
        float maxY = -numeric_limits<float>::infinity();

        for (const auto &pt : curr_vt_shape) {
            if (pt.zVal < minZ) minZ = pt.zVal;
            if (pt.zVal > maxZ) maxZ = pt.zVal;
            if (pt.yVal < minY) minY = pt.yVal;
            if (pt.yVal > maxY) maxY = pt.yVal;
        }

        // Create a a line that span from minZ to maxZ
        vector<float> line_zVal = utils.linspace(minZ-0.001, maxZ+0.001, 2);

        // Create the line_z
        line_z.resize(line_zVal.size(), {0.0f, 0.0f});

        // Compute Depth
        for(int cell_counter = 1; cell_counter<=air_cells_count; cell_counter++){

            // Find grid cell index for depthX and depthY computation
            int posY_depthX = lower_wallY - cell_counter;
            int posY_depthY = lower_wallY - cell_counter;

            int posY_depthX_idx = posY_depthX * frameW + posX;
            int posY_depthY_idx = posY_depthY * frameW + posX;

            // Find position along y-axis of the cross-section to compute depthX and depthY
            float curr_pos_depthX = (cell_counter-0.5)*ds;
            float curr_pos_depthY = cell_counter*ds;

            // Compute depthX
            // Step 1: Determine y coordinates of line_z
            // Step 2: Find the intersection between the cross-section and line
            // Step 3: Calculate difference between min and max z_intersection

            vector<float> line_yVal = utils.linspace(minY+curr_pos_depthX, minY+curr_pos_depthX, line_zVal.size());

            for(int idx=0; idx<line_zVal.size(); idx++){
                line_z[idx].yVal = line_yVal[idx];
                line_z[idx].zVal = line_zVal[idx];
            }

            vector<ShapeCoordinates> intersection_points = utils.findShapeIntersections(curr_vt_shape, line_z);

            // Find the min and max zVal of the cross-section
            float minZ_intersection =  numeric_limits<float>::infinity();
            float maxZ_intersection = -numeric_limits<float>::infinity();

            for (const auto &pt : intersection_points){
                if (pt.zVal < minZ_intersection) minZ_intersection = pt.zVal;
                if (pt.zVal > maxZ_intersection) maxZ_intersection = pt.zVal;
            }

            depthX_val = (maxZ_intersection - minZ_intersection)/2.0f;
            
            // Compute depthY
            if (cell_counter<air_cells_count){

                line_yVal = utils.linspace(minY+curr_pos_depthY, minY+curr_pos_depthY, line_zVal.size());

                for(int idx=0; idx<line_zVal.size(); idx++){
                    line_z[idx].yVal = line_yVal[idx];
                    line_z[idx].zVal = line_zVal[idx];
                }

                intersection_points = utils.findShapeIntersections(curr_vt_shape, line_z);

                // Find the min and max zVal of the cross-section
                minZ_intersection =  numeric_limits<float>::infinity();
                maxZ_intersection = -numeric_limits<float>::infinity();

                for (const auto &pt : intersection_points){
                    if (pt.zVal < minZ_intersection) minZ_intersection = pt.zVal;
                    if (pt.zVal > maxZ_intersection) maxZ_intersection = pt.zVal;
                }

                depthY_val = (maxZ_intersection - minZ_intersection)/2.0f;

            }else{
                depthY_val = 0;
            }

            // Print depthX and depthY to verify
            // cout << "cell_counter = " << cell_counter << " depthX = "<< depthX_val << " depthY = " << depthY_val << endl;
            
            // Throw error for nrgative depth value
            if(depthX_val < 0 || depthY_val< 0){
                cout << "Incorrect depth value: depthX = "<< depthX_val << " depthY = " << depthY_val << endl;
                exit(EXIT_FAILURE);
            }

            // Assign depthX and depthY to grid cells
            temp_depthX[posY_depthX_idx] = depthX_val;
            temp_depthY[posY_depthY_idx] = depthY_val;
        } 

        // Assign tube depth to zero for boundary cells
        int lowerY_idx = lower_wallY * frameW + posX;
        int upperY_idx = upper_wallY * frameW + posX;

        temp_depthX[lowerY_idx] = 0.0f;
        temp_depthX[upperY_idx] = 0.0f;
        temp_depthY[lowerY_idx] = 0.0f;
        temp_depthY[upperY_idx] = 0.0f;
    }                                                   
}

__global__
void resetOpenSpaceDepth(float* depthX, float* depthY, int midY, int tube_end_posX, int frameH, int frameW, const float open_space_depth){

    // Global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= frameH*frameW) return;

    int idx = midY * frameW + tube_end_posX;

    float mid_value_depthX = depthX[idx];
    float mid_value_depthY = depthY[idx];

    // Reset depthX
    if (depthX[tid] == open_space_depth)
        depthX[tid] = mid_value_depthX;

    // Reset depthY
    if (depthY[tid] == open_space_depth)
        depthY[tid] = mid_value_depthY;
}

// CUDA kernel to smooth each row of depthX
__global__
void smoothDepthX(
    const float* depthX,    
    float*       temp_depthX,   // where to write smoothed depthX
    int          frameH,
    int          frameW,
    int          midY,
    int          tubeEndX
){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = frameH*frameW;
    if (idx >= total) return;

    // int row = idx / frameW;
    int col = idx % frameW;

    float orig = depthX[idx];
    float right = (col < frameW-1)
        ? depthX[idx + 1]
        : depthX[midY*frameW + tubeEndX];      

    temp_depthX[idx] = 0.5f*(orig + right);
}

// CUDA kernel to smooth each column of depthY
__global__
void smoothDepthY(
    const float* depthY,    
    float*       temp_depthY,
    int          frameH,
    int          frameW,
    int          midY,
    int          tubeEndX
){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = frameH*frameW;
    if (idx >= total) return;

    int row = idx / frameW;
    // int col = idx % frameW;

    float orig = depthY[idx];
    float above = (row > 0)
        ? depthY[idx - frameW]
        : depthY[midY*frameW + tubeEndX];     

    temp_depthY[idx] = 0.5f*(orig + above);
}

__global__
void updateDepthP(float* depthP, float* depthX, float* depthY,
                  int frameH, int frameW, int midY, int tubeEndX){
    
    int total = frameH * frameW;
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int stride= blockDim.x * gridDim.x;
                
    // Pre‐compute the “open‐space pivot” index & values
    int midIdx = midY * frameW + tubeEndX;
    float pivotX = depthX[midIdx];
    float pivotY = depthY[midIdx];
    
    for (; tid < total; tid += stride) {
        int row = tid / frameW;   // 0-based row
        int col = tid % frameW;   // 0-based col
    
        // current
        float cX = depthX[tid];
        float cY = depthY[tid];
    
        // left neighbor in X
        float lX = (col > 0)
            ? depthX[ row*frameW + (col-1) ]
            : pivotX;
    
        // down neighbor in Y (“upDepthY” in MATLAB uses y+1)
        float dY = (row < frameH-1)
            ? depthY[(row+1)*frameW + col]
            : pivotY;
    
        // average
        depthP[tid] = 0.25f * (cX + cY + lX + dY);
    }
}

__global__
void setMinDepthVals(float* depthX, float* depthY, float* depthP, float min_depth, int frameH, int frameW){

    // Find the global thread ID
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;

    // 
    if( tid < frameH * frameW){
        if(depthP[tid] < min_depth) depthP[tid] = min_depth;
        if(depthX[tid] < min_depth) depthX[tid] = min_depth;
        if(depthY[tid] < min_depth) depthY[tid] = min_depth;
    }
}