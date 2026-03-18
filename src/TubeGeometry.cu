#include "../include/FDTDSolver.h"
#include "../include/TubeGeometry.h"


#define _USE_MATH_DEFINES

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
void setVocalTractWalls(FDTDSolver::GridCellComponents* d_PV_N, 
float* tube_cumm_segment_len,
int* tube_segment_diameter_incells,
int* boundary_segment_type, float ds,
int num_tube_segments, int tube_length_incells,
int* upper_wall_pos, int* lower_wall_pos,
int tube_start_posX, int tube_start_posY, int frameW);

__global__
void connectSagittalWallsGap(FDTDSolver::GridCellComponents* d_PV_N,
int* boundary_segment_type,
const int* upper_wall_pos,       
const int* lower_wall_pos,       
int tube_length_incells,  
int tube_start_posX,     
int frameH, int frameW);

__device__ __inline__
int findMaxInColumn(const int* boundary, int frameH, int frameW, int col);

__global__
void setExcitationCells(FDTDSolver::GridCellComponents* d_PV_N,
int tube_start_posX,
const int* upper_wall_pos,
const int* lower_wall_pos,
int frameW);

__global__
void setNoPressureCells(FDTDSolver::GridCellComponents* d_PV_N,
int tube_end_posX, int tube_length_incells,
const int* upper_wall_pos,
const int* lower_wall_pos,
int frameW);

// Read the tube geometry data
TubeGeometry::TubeGeometry(FDTDSolver& fdtdSolver):_fdtdSolver(fdtdSolver) {

    string file_name;

    // Read the tube geometry data: /data/tube-geometry/cylindrical-tube.txt
    if (_fdtdSolver.tube_geometry)
        if (_fdtdSolver.vowel_type == 1){
            cout << "Simulating vowel sound /a/" << endl;
            file_name = "vocaltract_a.txt";
        }else if (_fdtdSolver.vowel_type == 2) {
            cout << "Simulating vowel sound /i/" << endl;
            file_name = "vocaltract_i.txt";
        }else if (_fdtdSolver.vowel_type == 3) {
            cout << "Simulating vowel sound /u/" << endl;
            file_name = "vocaltract_u.txt";
        }
        
    else
        file_name = "cylindrical-tube.txt";

    string file_path = "../../data/tube-geometry/" +  file_name;

    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << file_name << endl;
        return;
    }
    else
        cerr << "Successfully opened file " << file_name << endl;

    float segment_length, cross_sectional_area;

    while (file >> segment_length >> cross_sectional_area) {
        tube_segment_length_incm.push_back(segment_length); 
        cross_sectional_area_incm2.push_back(cross_sectional_area);
    }

    // Number of tube segements
    num_tube_segments = tube_segment_length_incm.size();

    // Compute the tube segment diameter
    computeTubeWidth();

    // Convert the tube segment length to meter
    for(float segment_len : tube_segment_length_incm){
        float segment_len_inm = segment_len * CENTIMETER;
        tube_segment_length_inm.push_back(segment_len_inm);
    }

    file.close();

    // Set the microphone/listener position
    mic_position = 3 * MILLIMETER;
    mic_position_cells = static_cast<int>(round(mic_position/_fdtdSolver.ds));
}

/***************H O S T*************F U N C T I O N S***************/

// Determine the computational domain size
void TubeGeometry::setComputationalDomain(){

    // Calculate the total tube length
    tube_length = accumulate(tube_segment_length_inm.begin(), tube_segment_length_inm.end(), 0.0f);
    
    tube_length_incells = static_cast<int>(round(tube_length/_fdtdSolver.ds));
    
    // solver_type = 2.5D : We don't need area function scaling
    // solver_type = 2D   : we need area function scaling

    if (_fdtdSolver.fdtd_solver_type == SIMULATION_2_5D)
        tube_segment_diameter = tube_segment_diameter_inm;
    else
        tube_segment_diameter = tubeAreaScaling(tube_segment_diameter_inm);

    // Find tube segment diameter in terms of number of grid cells
    for(float segment_diameter : tube_segment_diameter){
        h_tube_segment_diameter_incells.push_back(static_cast<int>(round(segment_diameter/_fdtdSolver.ds)));
    }

    // Make sure segment diameter have odd number of grid cells
    for(size_t idx =0; idx < h_tube_segment_diameter_incells.size(); idx++){

        if (h_tube_segment_diameter_incells[idx] == 0){
            h_tube_segment_diameter_incells[idx]++;
        }

        if (h_tube_segment_diameter_incells[idx] % 2 == 0){

            float diff = static_cast<float>(h_tube_segment_diameter_incells[idx]) - (tube_segment_diameter[idx]/_fdtdSolver.ds);

            if (diff > 0 || _fdtdSolver.cross_sectional_shape>=3)
                h_tube_segment_diameter_incells[idx]--;
            else
                h_tube_segment_diameter_incells[idx]++;
        }
    }

    // Percentage error in estimating total tube length
    float estimated_tube_length = static_cast<float> (tube_length_incells) * _fdtdSolver.ds;
    float percentage_error = ((estimated_tube_length - tube_length)/tube_length) * 100;
    cout << "Actual tube length = " << tube_length <<" m" << endl;
    cout << "Estimated tube length = " << estimated_tube_length <<" m" << endl;
    cout << "Tube length percentage error = " << percentage_error << " %" << endl;

    // Apply mouth radiation condition
    if (_fdtdSolver.mouth_radiation_condition == 1){ // For Dirichlet condition

        // +1 = excitation, +1 Dirichlet condition
        _fdtdSolver.domainW = tube_length_incells + 1 + 1; 

        // +2 = tube walls
        _fdtdSolver.domainH = *max_element(h_tube_segment_diameter_incells.begin(), h_tube_segment_diameter_incells.end()) + 2;

    }
    else{ // For no-Dirichlet condition

        _fdtdSolver.domainW = tube_length_incells + 1 + mic_position_cells;
        _fdtdSolver.domainH = *max_element(h_tube_segment_diameter_incells.begin(), h_tube_segment_diameter_incells.end()) + 2;   
    }

    // Compute the computational domian size
    _fdtdSolver.frameW = _fdtdSolver.domainW + 2; // For dead cells
    _fdtdSolver.frameW = _fdtdSolver.frameW + (2 * _fdtdSolver.num_pml_layers * _fdtdSolver.pml_flag);
    
    _fdtdSolver.frameH = _fdtdSolver.domainH + 2; // For dead cells
    _fdtdSolver.frameH = _fdtdSolver.frameH + (2 * _fdtdSolver.num_pml_layers * _fdtdSolver.pml_flag);
}

// Generate sagittal vocal tract walls
void TubeGeometry::generateVocalTractWalls(){

    // Set thread configuration
    int NUM_THREADS = tube_length_incells;
    int NUM_THREADS_PER_BLOCK = 128;
    int NUM_BLOCKS = (NUM_THREADS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

    // Find the starting and ending position of the tube
    // Tube Start: 0 - For starting point | 1 - For cell_dead layer and excitation
    tube_start_posX = 0 + 1 + 1 + (_fdtdSolver.num_pml_layers * _fdtdSolver.pml_flag);
    tube_start_posY = static_cast<int>(ceil(_fdtdSolver.frameH/2.0f)) - 1;

    tube_end_posX = tube_start_posX + tube_length_incells - 1;
    tube_end_posY = static_cast<int>(ceil(_fdtdSolver.frameH/2.0f)) - 1;

    // Find the upper and lower wall positions
    cudaMalloc(&upper_wall_pos, tube_length_incells*sizeof(int));
    cudaMalloc(&lower_wall_pos, tube_length_incells*sizeof(int));

    // Initiate upper and lower wall positions to zero
    CHECK(cudaMemset(upper_wall_pos, 0, tube_length_incells*sizeof(int)));
    CHECK(cudaMemset(lower_wall_pos, 0, tube_length_incells*sizeof(int)));

    // Compute the cumulative length at the end of each tube segment
    thrust::device_vector<float> tube_cumm_segment_len(
        tube_segment_length_inm.begin(),
        tube_segment_length_inm.end()
    );

    // compute the in-place prefix sums on the device
    thrust::inclusive_scan(
        tube_cumm_segment_len.begin(),
        tube_cumm_segment_len.end(),
        tube_cumm_segment_len.begin()
    );

    // Create a device vector to store tube segments' diameters
    thrust::device_vector<int> d_tube_segment_diameter_incells = h_tube_segment_diameter_incells;
    
    // Generate 2D sagittal contours of the tube
    setVocalTractWalls<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(_fdtdSolver.PV_N.data()),
                                                              thrust::raw_pointer_cast(tube_cumm_segment_len.data()),
                                                              thrust::raw_pointer_cast(d_tube_segment_diameter_incells.data()),
                                                              _fdtdSolver.getBoundarySegmentType(), _fdtdSolver.ds,
                                                              num_tube_segments, tube_length_incells,
                                                              upper_wall_pos, lower_wall_pos,
                                                              tube_start_posX, tube_start_posY, _fdtdSolver.frameW);
    CHECK(cudaGetLastError());

    // Connect gaps between the adjacent vocal tract wall segments
    connectSagittalWallsGap<<<1, 1>>>(thrust::raw_pointer_cast(_fdtdSolver.PV_N.data()),
                                                               _fdtdSolver.getBoundarySegmentType(),
                                                               upper_wall_pos, lower_wall_pos,
                                                               tube_length_incells, tube_start_posX, 
                                                               _fdtdSolver.frameH, _fdtdSolver.frameW);
    CHECK(cudaGetLastError());
}

// Area function scaling for the 2D model
vector <float> TubeGeometry::tubeAreaScaling(vector<float> tube_segment_diameter_inm){

    // Find the max diameter and the corresponding index
    auto max_diameter = *max_element(tube_segment_diameter_inm.begin(), tube_segment_diameter_inm.end());
    size_t max_idx = 0;

    for(size_t idx=0; idx < tube_segment_diameter_inm.size(); idx++){

        if(tube_segment_diameter_inm[idx] == max_diameter)
            max_idx = idx;
    }

    // Initialize the tube_segment_diameter to zeros
    vector<float> tube_segment_diameter(tube_segment_diameter_inm.size(), 0.0f);

    // At max index position, d_2D = d3D(0.5*pi/1.84) 
    tube_segment_diameter[max_idx] = (max_diameter * 0.5 * M_PI)/1.84f;

    //Build the expansion ratio array m
    vector<float> m(tube_segment_diameter_inm.size(), 1.0f);

    for (size_t i = 0; i < tube_segment_diameter_inm.size(); ++i) {
        if (i < max_idx) {
            float a = tube_segment_diameter_inm[i];
            float b = tube_segment_diameter_inm[i + 1];
            m[i] = (b != 0.0f) ? std::pow(a / b, 2.0f) : 1.0f;
        }
        else if (i > max_idx) {
            float a = tube_segment_diameter_inm[i];
            float b = tube_segment_diameter_inm[i - 1];
            m[i] = (b != 0.0f) ? std::pow(a / b, 2.0f) : 1.0f;
        }
    }

    // Propagate backwards from max_idx−1 down to 0
    for (int i = int(max_idx) - 1; i >= 0; --i) {
        tube_segment_diameter[i] = tube_segment_diameter[i + 1] * m[i];
    }

    //  Propagate forwards from max_idx+1 up to n−1
    for (size_t i = max_idx + 1; i < tube_segment_diameter_inm.size(); ++i) {
        tube_segment_diameter[i] = tube_segment_diameter[i - 1] * m[i];
    }

    return tube_segment_diameter;
}

// Generate excitation cells
void TubeGeometry::generateExcitationCells(){

    setExcitationCells<<<1, 1>>>(thrust::raw_pointer_cast(_fdtdSolver.PV_N.data()),
                                tube_start_posX, upper_wall_pos, lower_wall_pos, _fdtdSolver.frameW);

}

// Generate no-pressurce cells to enforce Dirichlet boundary condition
void TubeGeometry::generateMouthEndCells(){
    setNoPressureCells<<<1, 1>>>(thrust::raw_pointer_cast(_fdtdSolver.PV_N.data()),
                                 tube_end_posX, tube_length_incells,
                                 upper_wall_pos, lower_wall_pos, _fdtdSolver.frameW);
}

// Compute vocal tract sagittal width
void TubeGeometry::computeTubeWidth(){

    if (_fdtdSolver.cross_sectional_shape == CIRCULAR)
        computeTubeWidthCircle();

    else if(_fdtdSolver.cross_sectional_shape == ELLIPTICAL)
        computeTubeWidthEllipse();

    else if(_fdtdSolver.cross_sectional_shape == SINGLE_PLANE_SYMMETRIC)
        computeTubeWidthSinglePlaneSymmetry();   
}

void TubeGeometry::computeTubeWidthCircle(){

    // Calculate diameter of tube segments - For circular cross-section
    for(float tubeArea : cross_sectional_area_incm2){

        // Convert the segment area into meter and save it in cross_sectional_area_inm2
        float segment_area_inm2 = tubeArea * CENTIMETER * CENTIMETER;
        cross_sectional_area_inm2.push_back(segment_area_inm2);

        // Compute the diameter from area
        float diameter = 2.0 * sqrt(segment_area_inm2 / M_PI);

        tube_segment_diameter_inm.push_back(diameter);
    }
}

void TubeGeometry::computeTubeWidthEllipse(){

    // Calculate semi-major and semi-minor axes - For elliptical cross-section
    // Set the semi major axis as the tube diameter
    for(float tubeArea : cross_sectional_area_incm2){

        // Convert the segment area into meter and save it in cross_sectional_area_inm2
        float segment_area_inm2 = tubeArea * CENTIMETER * CENTIMETER;
        cross_sectional_area_inm2.push_back(segment_area_inm2);

        // Set ratio of semi-major to semi-minor axis
        const float semi_major_axis = 3.0f;
        const float semi_minor_axis = 1.0f;
        
        // Find length of semi_major and semi_minor axis
        // if semimajorAxis:semiminorAxis = a:b then their length can be
        // semimajorAxisLen = ax,  semiminorAxisLen = bx
        // ellipseArea = pi*ax*bx;

        // x = sqrt(ellipseArea/(semimajorAxisLen*semiminorAxisLen))

        float lenX = sqrtf(segment_area_inm2/(semi_major_axis * semi_minor_axis * M_PI));

        float major_axis_len = lenX * (semi_major_axis*2);
        float minor_axis_len = lenX * (semi_minor_axis*2);

        major_axis_length_inm.push_back(major_axis_len);
        minor_axis_length_inm.push_back(minor_axis_len);

        // Setting the major axis length of the ellipse as the tube segment diameter.
        tube_segment_diameter_inm.push_back(major_axis_len);
    }     
}

void TubeGeometry::computeTubeWidthSinglePlaneSymmetry(){

    // For a given cross-sectional area, generate two regular intersected shapes
    // Intersected shapes could be - circle-circle, circle-ellipse, and ellipse-ellipse
    // These geometries are mirror-symmetric. Therefore, we define a portion of the cross-section
    // to derive the tube width along the y-axis and the tube depth.
        
    // Number of coordinates to generate a half of the tube cross-section
    int num_coordinates = 1999;

    // Create an array to save coordinates of each cross-sectional shape
    tube_shapes.resize(num_tube_segments);
    aligned_tube_shapes.resize(num_tube_segments); 

    // For circular shape, set semiMajorAxis:semiMinorAxis = 1:1;
    // For elliptical shape, set semiMajorAxis:semiMinorAxis = 3:1;

    vector<float> s1_semi_major_axis_len;
    vector<float> s1_semi_minor_axis_len;
    vector<float> s2_semi_major_axis_len;
    vector<float> s2_semi_minor_axis_len;
    vector<float> s1_s2_center_distance;
        
    vector<float> yIntersect;
    vector<float> zIntersect;

    // Create an instance of ShapeGenerator
    ShapeGenerator* shapeGenerator;
    shapeGenerator = new ShapeGenerator(
        num_tube_segments,
        s1_semi_major_axis_len, s1_semi_minor_axis_len,
        s2_semi_major_axis_len, s2_semi_minor_axis_len,
        s1_s2_center_distance, yIntersect, zIntersect,
        0.0f
    );

    int section_idx = 0;
    
    for(float tubeArea : cross_sectional_area_incm2){

        // Convert the segment area into meter and save it in cross_sectional_area_inm2
        float segment_area_inm2 = tubeArea * CENTIMETER * CENTIMETER;
        cross_sectional_area_inm2.push_back(segment_area_inm2);

        // Generate the contour shape using the segment area
        shapeGenerator->segment_area = segment_area_inm2;
        shapeGenerator->combineRegularShapes(section_idx, num_coordinates);
        
        // Save the tube shape
        tube_shapes[section_idx] = shapeGenerator->full_shape_coordinates;
        aligned_tube_shapes[section_idx] = shapeGenerator->aligned_full_shape_coordinates;

        // Save the tube diameter
        tube_segment_diameter_inm.push_back(shapeGenerator->segment_diameter);
        
        // Increment the section index
        section_idx++;
    }

    // Save the tube segment coordinates
    Utils utils;
    utils.saveTubeSegmentCoords(tube_shapes, tube_segment_length_incm);
}

/**************C U D A***K E R N E L***F U N C T I O N S*************/

// Global function to set the vocal tract walls
__global__
void setVocalTractWalls(FDTDSolver::GridCellComponents* d_PV_N, 
                        float* tube_cumm_segment_len,
                        int* tube_segment_diameter_incells,
                        int* boundary_segment_type, float ds,
                        int num_tube_segments, int tube_length_incells,
                        int* upper_wall_pos, int* lower_wall_pos,
                        int tube_start_posX, int tube_start_posY, int frameW){

    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tube_length_incells) return;

    // Compute the which cell we are at along the tube length
    int tube_cell_counter = idx + 1;
    float curr_len = tube_cell_counter * ds;
    
    // Half grid cell length
    float half_Cell_len = ds * 0.5f;

    // Define the segment counter
    int segment_counter = 0;

    // scan until this cumulative length exceeds curr_len
    while (segment_counter < num_tube_segments - 1 && curr_len > tube_cumm_segment_len[segment_counter] + half_Cell_len) {
        ++segment_counter;
    }

    // Find the radius (in cells)
    int radius = (tube_segment_diameter_incells[segment_counter] - 1) / 2;

    // Compute the upper and lower wall positions
    int x = tube_start_posX + idx; 
    int y_up   = tube_start_posY - radius - 1;
    int y_down = tube_start_posY + radius + 1;

    // Record them in the upper/lower wall arrays
    upper_wall_pos[idx] = y_up;
    lower_wall_pos[idx] = y_down;

    // Flattened index
    int up_idx   = y_up   * frameW + x;
    int down_idx = y_down * frameW + x;

    // Set the cell types to cell_wall
    d_PV_N[up_idx].cell_type   = cell_wall;
    d_PV_N[down_idx].cell_type = cell_wall;

    // Record segment number
    boundary_segment_type[up_idx]   = segment_counter + 1;
    boundary_segment_type[down_idx] = segment_counter + 1;
}

__device__ __inline__
int findMaxInColumn(const int* boundary_segment_type, int frameH, int frameW, int col) {
    int m = 0;
    for(int y = 0; y < frameH; ++y) {
        int idx = y*frameW + col;
        m = max(m, boundary_segment_type[idx]);
    }
    return m;
}

__global__
void connectSagittalWallsGap(FDTDSolver::GridCellComponents* d_PV_N,
                             int* boundary_segment_type,
                             const int* upper_wall_pos,
                             const int* lower_wall_pos,
                             int tube_length_incells,
                             int tube_start_posX,
                             int frameH,
                             int frameW){
                                
    // Find the previous wall positions
    int prev_upper_wallY = upper_wall_pos[0];
    int prev_upper_wallX = tube_start_posX;
    int prev_lower_wallY = lower_wall_pos[0];
    int prev_lower_wallX = tube_start_posX;

    for(int tube_cell_counter=1; tube_cell_counter<=tube_length_incells-2; tube_cell_counter++){

        // Find the current wall positions
        int curr_upper_wallY = upper_wall_pos[tube_cell_counter];
        int curr_upper_wallX = tube_start_posX + tube_cell_counter;
        int curr_lower_wallY = lower_wall_pos[tube_cell_counter];
        int curr_lower_wallX = tube_start_posX + tube_cell_counter;

        int wall_cells_diff = curr_upper_wallY - prev_upper_wallY;

        if(wall_cells_diff>1){

            int seg_up = findMaxInColumn(boundary_segment_type, frameH, frameW, curr_upper_wallX);
    
            // For upper walls
            for(int wall_counter = prev_upper_wallY; wall_counter<=curr_upper_wallY; wall_counter++){
                // Find idx for flattened out array 
                int idx = wall_counter * frameW + curr_upper_wallX;
                d_PV_N[idx].cell_type = cell_wall;
                boundary_segment_type[idx] = seg_up;
            }
    
            // For lower walls
            for (int wall_counter =curr_lower_wallY; wall_counter<=prev_lower_wallY; wall_counter++){
                // Find idx for flattened out array 
                int idx = wall_counter * frameW + curr_lower_wallX;
                d_PV_N[idx].cell_type = cell_wall;
                boundary_segment_type[idx] = seg_up;
            }
        }
        else if(wall_cells_diff<-1){
    
            int seg_down = findMaxInColumn(boundary_segment_type, frameH, frameW, prev_upper_wallX);
    
            // For upper wall
            for(int wall_counter=curr_upper_wallY; wall_counter <= prev_upper_wallY; wall_counter++){
                // Find idx for flattened out array 
                int idx = wall_counter * frameW + prev_upper_wallX;
                d_PV_N[idx].cell_type = cell_wall;
                boundary_segment_type[idx] = seg_down;
            }
    
            // For lower wall
            for(int wall_counter=prev_lower_wallY; wall_counter<=curr_lower_wallY; wall_counter++){
                // Find idx for flattened out array 
                int idx = wall_counter * frameW + prev_lower_wallX;
                d_PV_N[idx].cell_type = cell_wall;
                boundary_segment_type[idx] = seg_down;
            }
        }

        // Assign the current wall positions to previous wall positions
        prev_upper_wallY = curr_upper_wallY;
        prev_upper_wallX = curr_upper_wallX;
        prev_lower_wallY = curr_lower_wallY;
        prev_lower_wallX = curr_lower_wallX;
    }    
}

__global__
void setExcitationCells(FDTDSolver::GridCellComponents* d_PV_N,
                        int tube_start_posX,
                        const int* upper_wall_pos,
                        const int* lower_wall_pos,
                        int frameW){

    int excitation_col = tube_start_posX - 1;

    for(int cell_counter = upper_wall_pos[0]+1; cell_counter<lower_wall_pos[0]; cell_counter++){

        // Find the idx for the flattened out array
        int idx = cell_counter * frameW + excitation_col;
        d_PV_N[idx].cell_type = cell_excitation;
    }

    // Set cells above and below the excitation column as wall cells
    int upper_idx = upper_wall_pos[0] * frameW + excitation_col;
    int lower_idx = lower_wall_pos[0] * frameW + excitation_col;

    d_PV_N[upper_idx].cell_type = cell_wall;
    d_PV_N[lower_idx].cell_type = cell_wall;

}

__global__
void setNoPressureCells(FDTDSolver::GridCellComponents* d_PV_N,
                        int tube_end_posX, int tube_length_incells,
                        const int* upper_wall_pos,
                        const int* lower_wall_pos,
                        int frameW){

    int mouth_end_col = tube_end_posX + 1;

    for(int cell_counter = upper_wall_pos[tube_length_incells-1]; cell_counter<=lower_wall_pos[tube_length_incells-1]; cell_counter++){

        // Find the idx for the flattened out array
        int idx = cell_counter * frameW + mouth_end_col;
        d_PV_N[idx].cell_type = cell_noPressure;
    }

}

int TubeGeometry::getFrameH(){return _fdtdSolver.frameH;}
int TubeGeometry::getFrameW(){return _fdtdSolver.frameW;}
int TubeGeometry::getVocalTractShape(){return _fdtdSolver.cross_sectional_shape;}
int TubeGeometry::getTubeLengthCells(){return tube_length_incells;}
float TubeGeometry::getSpatialResolution(){return _fdtdSolver.ds;}
int* TubeGeometry::getBoundarySegmentType(){return _fdtdSolver.getBoundarySegmentType();}
vector<float> TubeGeometry::getMinorAxisLength(){return minor_axis_length_inm;}
vector<float> TubeGeometry::getTubeSegmentLengthInCM(){return tube_segment_length_incm;}