#include "../include/FDTDSolver.h"
#include "../include/DeviceSymbols.h"
#include "../include/TubeGeometry.h"


// Check error codes for CUDA functions
#define CHECK(call){                                           \
    cudaError_t error = call;                                  \
    if (error != cudaSuccess){                                 \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error,       \
        cudaGetErrorString(error));                            \
    }                                                          \
}

// Kernel function declarations
__global__ void setupDeadCells(FDTDSolver::GridCellComponents* d_PV_N, 
                                        int frameH, 
                                        int frameW, 
                                        int cell_dead);
                            
__global__ void setupHorizontalPMLLayers(FDTDSolver::GridCellComponents* d_PV_N,
                                        int cell_pos1,
                                        int cell_pos2,
                                        int cell_type);
                                
__global__ void setupVerticalPMLLayers(FDTDSolver::GridCellComponents* d_PV_N,
                                       int frameW,
                                       int y_start,
                                       int y_end,
                                       int col_left,    
                                       int cell_type);
                                
__global__ void extractCellTypesBetaVals(FDTDSolver::GridCellComponents* d_PV_N, int num_interior_cells, int frameW, 
                                         int* curr_cell_type, int* right_cell_type, int* top_cell_type,
                                         float* curr_beta, float* right_beta, float* top_beta);

__global__ void extractNormalComponents(int num_interior_cells, float* d_N_out_val, float* d_N_in_val, 
                                        float* d_cur_beta, float* d_right_beta, float* d_top_beta);

__global__ void extractSigmaPrimedt(int num_interior_cells, int* curr_cell_type, int* right_cell_type, int* top_cell_type,
                                    float* d_sigma_prime_dt_cur, float* d_sigma_prime_dt_right, float* d_sigma_prime_dt_top);
                                
__global__ void isExcitationCell(int num_interior_cells, int* curr_cell_type, int* right_cell_type, int* top_cell_type,
                                 int* is_excitation_cell_cur, int* is_excitation_cell_right, int* is_excitation_cell_top);

__global__ void setMinVxVyBeta(int num_interior_cells, float* d_cur_beta, float* d_right_beta, 
                               float* d_top_beta, float* minVxBeta, float* minVyBeta);

__global__ void setMaxVxVySigmaPrimedt(int num_interior_cells, float* d_sigma_prime_dt_cur, float* d_sigma_prime_dt_right, 
                                       float* d_sigma_prime_dt_top, float* maxVxSigmaPrimedt, float* maxVySigmaPrimedt);

__global__ void setBetaVxVyParams(int num_interior_cells, float* minVxBeta, float* minVyBeta, float* betaVxSqr, float* betaVySqr, 
                                  float* betaVxSqr_dt_inv_rho_inv_ds, float* betaVySqr_dt_inv_rho_inv_ds);

__global__ void setAreWeNotExcitationVxVy(int num_interior_cells, int* is_excitation_cell_cur, int* is_excitation_cell_right, int* is_excitation_cell_top,
                                          float* areWeNotExcitationVx, float* areWeNotExcitationVy);

__global__ void setExcitationWeightVxVy(int num_interior_cells, int* is_excitation_cell_cur, int* is_excitation_cell_right, int* is_excitation_cell_top,
                                        float* excitationWeightVx, float* excitationWeightVy);
                                        
__global__ void setXORVals(int num_interior_cells, float* d_cur_beta, float* d_right_beta, float* d_top_beta,
                           float* xor_val1, float* xor_val2, float* xor_val3, float* xor_val4);                                       

__device__ int betaLookUpTable(int beta_right, int beta_top);

__global__
void setZInverse(float* z_inv, int* boundary_segment_type, 
int frameH, int frameW, float z_inv_val);

FDTDSolver::FDTDSolver(SimulationParams simulationParams){

    fdtd_solver_type          = simulationParams.getFDTDSolverType();
    simulation_type           = simulationParams.getSimulationType();
    tube_geometry             = simulationParams.getTubeGeometry();
    vowel_type                = simulationParams.getVowelType();
    cross_sectional_shape     = simulationParams.getCrossSectionalShape();
    mouth_radiation_condition = simulationParams.getMouthRadiationCond();
    audio_dur                 = simulationParams.getAudioDur();   
    source_type               = simulationParams.getSourceType(); // 0: Sine wave 1: Gaussian 2: Impulse 3: Two-mass model 4: Reed model
    pml_flag                  = simulationParams.getPMLFlag();    // 0: OFF 1: ON
    srate_mul                 = simulationParams.getSrateMul();
}

/***************C O D E***********V A L I D A T I O N***************/

// Print solver coefficient in matrix format to a CSV file
void FDTDSolver::printInMatrixFormat(float* data, string file_name, int num_rows, int num_cols){

    int num_interior_cells = num_rows * num_cols;

    // Allocate host memory
    float* h_data = new float[num_interior_cells];

    // Copy data from device to host
    CHECK(cudaMemcpy(h_data, data, num_interior_cells * sizeof(float), cudaMemcpyDeviceToHost));

    // File path
    string file_path = "../../data/" + file_name + ".csv";

    // Write data to CSV file
    std::ofstream outFile(file_path, std::ios::out | std::ios::trunc);
    if (outFile) {
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                int idx = i * num_cols + j;
                outFile << h_data[idx];
                if (j < frameW - 1) outFile << ",";
            }
            outFile << "\n";
        }
        outFile.close();
    } else {
        std::cerr << "Failed to open " << file_name+".csv" << " for writing." << std::endl;
    }
}

void FDTDSolver::printInMatrixFormat(int* data, string file_name, int num_rows, int num_cols){

    int num_interior_cells = num_rows * num_cols;

    // Allocate host memory
    int* h_data = new int[num_interior_cells];

    // Copy data from device to host
    CHECK(cudaMemcpy(h_data, data, num_interior_cells * sizeof(int), cudaMemcpyDeviceToHost));

    // File path
    string file_path = "../../data/" + file_name + ".csv";

    // Write data to CSV file
    std::ofstream outFile(file_path, std::ios::out | std::ios::trunc);
    if (outFile) {
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                int idx = i * num_cols + j;
                outFile << h_data[idx];
                if (j < frameW - 1) outFile << ",";
            }
            outFile << "\n";
        }
        outFile.close();
    } else {
        std::cerr << "Failed to open " << file_name+".csv" << " for writing." << std::endl;
    }
}

// Print the grid cell components (P, Vx, Vy or cell_types) to a CSV file
void FDTDSolver::printGridCellComponents() {

    // Copy device_vector to host_vector
    thrust::host_vector<GridCellComponents> host_PV_N = PV_N;

    // Lambda function to write *any* field into a CSV
    auto writeField = [&](auto fieldGetter, const std::string &filename) {
        std::ofstream outFile(filename, std::ios::out | std::ios::trunc);
        if (!outFile) {
            std::cerr << "Failed to open " << filename << "\n";
            return;
        }
        for (int i = 0; i < frameH; ++i) {
            for (int j = 0; j < frameW; ++j) {
                int idx = i * frameW + j;
                outFile << fieldGetter(host_PV_N[idx]);
                if (j + 1 < frameW) outFile << ',';
            }
            outFile << '\n';
        }
        outFile.close();
    };

    // Call it for each component:
    writeField(
      [](const GridCellComponents &c){ return c.Pr; },
      "../../data/Pressure.csv"
    );
    writeField(
      [](const GridCellComponents &c){ return c.Vx; },
      "../../data/Vx.csv"
    );
    writeField(
      [](const GridCellComponents &c){ return c.Vy; },
      "../../data/Vy.csv"
    );
    writeField(
      [](const GridCellComponents &c){ return c.cell_type; },
      "../../data/cell_types.csv"
    );
}

// Print depthComponents (depthP, depthX, depthY) to a CSV file
void FDTDSolver::printDepthComponents(float* data, string file_name){

    // Calculate the number of cells in the frame
    int num_cells = frameH * frameW;

    // Allocate host memory
    float* h_data = new float[num_cells];

    // Copy data from device to host
    CHECK(cudaMemcpy(h_data, data, num_cells * sizeof(float), cudaMemcpyDeviceToHost));

    // File path to copy data
    string file_path = "../../data/" + file_name + ".csv";

    // Write data to CSV file
    std::ofstream outFile(file_path, std::ios::out | std::ios::trunc);
    if (outFile) {
        for (int i = 0; i < frameH; ++i) {
            for (int j = 0; j < frameW; ++j) {
                int idx = i * frameW + j;
                outFile << h_data[idx];
                if (j + 1 < frameW) outFile << ",";
            }
            outFile << "\n";
        }
        outFile.close();
    } else {
        std::cerr << "Failed to open " << file_name+".csv" << " for writing." << std::endl;
    }
}

// Print the current and neighboring cells data to a CSV file
void FDTDSolver::printNeighboringCellsData(string file_name, float* data_cur, float* data_right, float* data_top){

    int num_interior_cells = (frameH-2) * (frameW-2);

    // File path to copy data
    string file_path = "../../data/" + file_name + ".csv";

    // Allocate host memory
    thrust::host_vector<float> h_data_cur(num_interior_cells);
    thrust::host_vector<float> h_data_right(num_interior_cells);
    thrust::host_vector<float> h_data_top(num_interior_cells);

    // Copy data from device to host
    CHECK(cudaMemcpy(h_data_cur.data(), data_cur, num_interior_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_data_right.data(), data_right, num_interior_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_data_top.data(), data_top, num_interior_cells * sizeof(float), cudaMemcpyDeviceToHost));

    // Write values to a csv file
    std::ofstream outFile(file_path, std::ios::out | std::ios::trunc);

    if (outFile) {
        outFile << "cuurent, right, top\n";
        for (int i = 0; i < num_interior_cells; ++i) {
            outFile << h_data_cur[i] << "," << h_data_right[i] << "," << h_data_top[i] << "\n";
        }
        outFile.close();
    } else {
        std::cerr << "Failed to open " << file_name+".csv" << " for writing." << std::endl;
    }
}

void FDTDSolver::printNeighboringCellsData(string file_name, int* data_cur, int* data_right, int* data_top){

    int num_interior_cells = (frameH-2) * (frameW-2);

    // File path to copy data
    string file_path = "../../data/" + file_name + ".csv";

    // Allocate host memory
    thrust::host_vector<int> h_data_cur(num_interior_cells);
    thrust::host_vector<int> h_data_right(num_interior_cells);
    thrust::host_vector<int> h_data_top(num_interior_cells);

    // Copy data from device to host
    CHECK(cudaMemcpy(h_data_cur.data(), data_cur, num_interior_cells * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_data_right.data(), data_right, num_interior_cells * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_data_top.data(), data_top, num_interior_cells * sizeof(int), cudaMemcpyDeviceToHost));

    // Write values to a csv file
    std::ofstream outFile(file_path, std::ios::out | std::ios::trunc);

    if (outFile) {
        outFile << "current, right, top\n";
        for (int i = 0; i < num_interior_cells; ++i) {
            outFile << h_data_cur[i] << "," << h_data_right[i] << "," << h_data_top[i] << "\n";
        }
        outFile.close();
    } else {
        std::cerr << "Failed to open " << file_name+".csv" << " for writing." << std::endl;
    }
}

/***************H O S T*************F U N C T I O N S***************/

// Set up the wave solver using the simulation parameters
void FDTDSolver::setupWaveSolver(){
    
    // Solver constants
    open_space_depth = 0.50;

    // Set up the FDTD wave solver using the simulation parameters
    // Transfer parameters to the device
    SimulationParams simulationParams(srate_mul);

    sound_speed = simulationParams.sound_speed;
    rho = simulationParams.rho;
    ds = simulationParams.ds;
    dt = simulationParams.dt;
    wall_pressure_coupling_coeff = simulationParams.wall_pressure_coupling_coeff;
    Mw = simulationParams.Mw;
    Bw = simulationParams.Bw;
    Kw = simulationParams.Kw;

    kappa            = simulationParams.kappa;
    max_sigma_dt     = simulationParams.max_sigma_dt;
    num_pml_layers   = simulationParams.pml_layers;

    rho_sqrC_dt_inv_ds = (kappa*simulationParams.dt)/simulationParams.ds;
    rho_sqrC_dt = kappa * simulationParams.dt;

    // Copy the simulation parameters to device
    CHECK(cudaMemcpyToSymbol(d_sound_speed, &sound_speed, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_rho, &rho, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_ds, &ds, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_dt, &dt, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_wall_pressure_coupling_coeff, &wall_pressure_coupling_coeff, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_Mw, &Mw, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_Bw, &Bw, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_Kw, &Kw, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_rho_sqrC_dt_inv_ds, &rho_sqrC_dt_inv_ds, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_rho_sqrC_dt, &rho_sqrC_dt, sizeof(float)));

    // Set up the FDTD wave solver using the solver parameters
    SolverParams solverParams;
    
    solverParams.setBeta(num_pml_layers);
    solverParams.setSigmaPrimeDt(max_sigma_dt, simulationParams.dt, num_pml_layers);
    solverParams.setSourceDirection();

    beta = solverParams.getBeta();
    sigma_prime_dt = solverParams.getSigmaPrimeDt();
    source_direction = solverParams.getSourceDirection();

    // Copy beta, sigma_prime_dt and source_direction to device
    CHECK(cudaMemcpyToSymbol(d_beta, beta, NUM_CELL_TYPES * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_sigma_prime_dt, sigma_prime_dt, NUM_CELL_TYPES * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_source_direction, source_direction, 4 * sizeof(float)));

    // Find the number of simulation steps
    solverParams.setTimeStep(audio_dur, simulationParams.dt);
    time_step = solverParams.getTimeStep();
    num_steps = time_step.size();
    
    // Set the domain for open space simulation
    if (simulation_type == 0)
        openSpaceSimulation();
    else
        tubeGeometrySimulation();
    

    // Set up dead cells around the domain  
    // Thread configuration
    int NUM_THREADS = frameH * frameW;
    int NUM_THREADS_PER_BLOCK = 256;
    int NUM_BLOCKS = (NUM_THREADS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

    setupDeadCells<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(PV_N.data()), 
                                                         frameH, 
                                                         frameW, 
                                                         cell_dead);

    // Set up PML layers around the domain
    // Create two streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int pml_cell_type = 0;

    if (pml_flag == 1){

        //Set horizontal pml layers

        pml_cell_type = cell_pml5;
        int y_shift = 1;
        int x_start = 1;
        int x_end = frameW - 2;

        int row_upper = 0;
        int row_lower = 0;
        int left_most_col = 0;
        int right_most_col = 0;

        for (int pml_counter = 0; pml_counter < num_pml_layers; pml_counter++){

            // Define the rows for which PML layers need to be set up
            row_upper = y_shift + pml_counter;
            row_lower = frameH - 1 - y_shift - pml_counter;

            // Define the left-most and right-most columns within which PML layers need to be set up
            left_most_col = row_upper * frameW + x_start;
            right_most_col = row_upper * frameW + x_end;

            setupHorizontalPMLLayers<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream1>>>(thrust::raw_pointer_cast(PV_N.data()),
                                                                                        left_most_col, 
                                                                                        right_most_col, 
                                                                                        pml_cell_type);

            left_most_col = row_lower * frameW + x_start;
            right_most_col = row_lower * frameW + x_end;

            setupHorizontalPMLLayers<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream2>>>(thrust::raw_pointer_cast(PV_N.data()),
                                                                                        left_most_col, 
                                                                                        right_most_col, 
                                                                                        pml_cell_type);

            x_start = x_start + 1;
            x_end = x_end - 1;
            pml_cell_type = pml_cell_type-1;
        }

        // Synchronize the device to ensure all kernels are completed
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        
        //Set vertical pml layers

        pml_cell_type = cell_pml5;

        int x_shift = 1;
        int y_start = 1;
        int y_end = frameH - 2;
        

        int col_left = 0;
        int col_right = 0;

        for (int pml_counter = 0; pml_counter < num_pml_layers; pml_counter++){

            // Define the columns for which PML layers need to be set up
            col_left = x_shift + pml_counter;
            col_right = frameW - 1 - x_shift - pml_counter;

            setupVerticalPMLLayers<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream1>>>(thrust::raw_pointer_cast(PV_N.data()),
                                                                                      frameW,
                                                                                      y_start,
                                                                                      y_end,
                                                                                      col_left,    
                                                                                      pml_cell_type);

            setupVerticalPMLLayers<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream2>>>(thrust::raw_pointer_cast(PV_N.data()),
                                                                                      frameW,
                                                                                      y_start,
                                                                                      y_end,
                                                                                      col_right,    
                                                                                      pml_cell_type);
            
            y_start = y_start + 1;
            y_end = y_end - 1;
            pml_cell_type = pml_cell_type-1;
        }

        // Synchronize the device to ensure all kernels are completed
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        
        // Destroy the streams
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
    }

    // Setup the source model
    GlottalExcitations glottalExcitations;
    glottalExcitations.setExcitationVelocity(source_type, num_steps, simulationParams.dt);
    thrust::host_vector<float> host_excitation_velocity = glottalExcitations.getExcitationVelocity();

    // Copy the excitation velocity from host to device
    excitation_velocity = host_excitation_velocity;  
    
    // calculate solver coefficients
    calculateSolverCoefficients();
}

// Set up the computational domain for the open space simulation
void FDTDSolver::openSpaceSimulation(){

    // Define number domain height and width
    domainH = 43;
    domainW = 341;

    // frame size = domain size + dead_cell layer + pml layer [Top/Bottom + Left/Right]
    frameH = domainH + 2 + (2*pml_flag*num_pml_layers); // Number of rows
    frameW = domainW + 2 + (2*pml_flag*num_pml_layers); // Number of columns

    cout << "Simulation frame height = " << frameH << endl;
    cout << "Simulation frame width = " << frameW << endl;

    // Resize PV_N and PV_Nplus1 to match frameH * frameW
    PV_N.resize(frameH * frameW);
    PV_Nplus1.resize(frameH * frameW);

    // Initiate acoustic parameters in each grid cell in PV_N and PV_Nplus1 to zero
    // Pr = 0, Vx = 0, Vy = 0, cell_type = cell_air
    GridCellComponents initValue = {0.0f, 0.0f, 0.0f, cell_air}; 
    thrust::fill(PV_N.begin(), PV_N.end(), initValue);
    thrust::fill(PV_Nplus1.begin(), PV_Nplus1.end(), initValue);

    // Resize boundary_segment_type and z_inv to match frameH * frameW
    cudaMalloc(&boundary_segment_type, (frameH) * (frameW) * sizeof(int));
    cudaMalloc(&z_inv, (frameH) * (frameW) * sizeof(float));
    
    // For openSpaceSimulation, there are no vocal tract walls/boundaries inside the computational domain.
    // Hence, boundary_segment_type and z_inv needs to be initialized to zero.
    CHECK(cudaMemset(boundary_segment_type, 0, (frameH) * (frameW) * sizeof(int)));
    CHECK(cudaMemset(z_inv, 0, (frameH) * (frameW) * sizeof(float)));

    // Resize depth parameters to match frameH * frameW
    depthP.resize(frameH * frameW);
    depthX.resize(frameH * frameW);
    depthY.resize(frameH * frameW);

    // Set up the depth parameters
    if (fdtd_solver_type == SIMULATION_2_5D){
        thrust::fill(depthP.begin(), depthP.end(), 1.0*open_space_depth);
        thrust::fill(depthX.begin(), depthX.end(), 1.0*open_space_depth);
        thrust::fill(depthY.begin(), depthY.end(), 1.0*open_space_depth);
    }
    else{
        thrust::fill(depthP.begin(), depthP.end(), 1.0);
        thrust::fill(depthX.begin(), depthX.end(), 1.0);
        thrust::fill(depthY.begin(), depthY.end(), 1.0);
    }

    // Set the midpoint of the frame as excitation cell
    // -1 as C++ indexing starts from 0
    int excitationY = ceil(frameH / 2) - 1; 
    int excitationX = ceil(frameW / 2) - 1;

    // Set the listener position - 5 cells away from the excitation cell
    listenerY = excitationY + 5;
    listenerX = excitationX + 5;

    cout<< "excitationY = " << excitationY << " excitationX = " << excitationX << endl;

    // Calculate the 1D index of the excitation cell
    int excitationCellIndex = (excitationY * frameW) + excitationX;

    // Create a host vector to modify PV_N on the host
    // Update the cell_type of the excitation cell and copy it back to the device vector
    thrust::host_vector<GridCellComponents> host_PV_N = PV_N;
    host_PV_N[excitationCellIndex].cell_type = cell_excitation;
    PV_N = host_PV_N;

    // Free unused global memory
    cudaFree(boundary_segment_type);
}

// Set up the computational domain for the tube geometry simulation
void FDTDSolver::tubeGeometrySimulation(){

    TubeGeometry tubeGeometry(*this);

    // Set up the computational domain [frameW, frameH] size
    tubeGeometry.setComputationalDomain();

    cout << "Simulation frame height = " << frameH << endl;
    cout << "Simulation frame width = " << frameW << endl;

    // Resize PV_N and PV_Nplus1 to match frameH * frameW
    PV_N.resize(frameH * frameW);
    PV_Nplus1.resize(frameH * frameW);

    // Initiate acoustic parameters in each grid cell in PV_N and PV_Nplus1 to zero
    // Pr = 0, Vx = 0, Vy = 0, cell_type = cell_air
    GridCellComponents initValue = {0.0f, 0.0f, 0.0f, cell_air}; 
    thrust::fill(PV_N.begin(), PV_N.end(), initValue);
    thrust::fill(PV_Nplus1.begin(), PV_Nplus1.end(), initValue);

    // Resize boundary_segment_type and z_inv to match frameH * frameW
    cudaMalloc(&boundary_segment_type, (frameH) * (frameW) * sizeof(int));
    cudaMalloc(&z_inv, (frameH) * (frameW) * sizeof(float));
    
    // Initialize boundary_segment_type and z_inv to zero.
    CHECK(cudaMemset(boundary_segment_type, 0, (frameH) * (frameW) * sizeof(int)));
    CHECK(cudaMemset(z_inv, 0, (frameH) * (frameW) * sizeof(float)));

    // Generate vocal tract walls
    tubeGeometry.generateVocalTractWalls();

    // Set excitation cells
    tubeGeometry.generateExcitationCells();

    // Validate the boundary_segment_type
    // printInMatrixFormat(boundary_segment_type, "boundary_segment_type", frameH, frameW);

    // Set no_pressure cells to enforce Dirichlet boundary condition
    if(this->mouth_radiation_condition){

        tubeGeometry.generateMouthEndCells();
        
        // Find listener postion
        listenerX = tubeGeometry.tube_end_posX - tubeGeometry.mic_position_cells;
        listenerY = tubeGeometry.tube_end_posY;

        // Find excitation position
        excitationX = tubeGeometry.tube_start_posX - 1;
        excitationY = listenerY;
    }
    else{
        // Find listener postion
        listenerX = tubeGeometry.tube_end_posX + 3 + tubeGeometry.mic_position_cells;
        listenerY = tubeGeometry.tube_end_posY;

        // Find excitation position
        excitationX = tubeGeometry.tube_start_posX - 1;
        excitationY = listenerY;
    }

    // Set semi-reflective boundary impedance using z_inv
    float mu_3D = (fdtd_solver_type == SIMULATION_2_5D) ? 0.025f : 0.005f;
    float mu_2D = mu_3D * ((2.0f * 0.5F * M_PI)/1.84f);
    float alpha = 1.0f/(0.5f +0.25f *(mu_2D +(1/mu_2D)));
    float z_inv_val = 1 / (rho * sound_speed * ( (1+sqrt(1-alpha))/(1-sqrt(1-alpha))));
    
    int NUM_THREADS = frameH * frameW;
    int NUM_THREADS_PER_BLOCK = 256;
    int NUM_BLOCKS = (NUM_THREADS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    setZInverse<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(z_inv, boundary_segment_type, frameH, frameW, z_inv_val);

    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Resize depth parameters to match frameH * frameW
    depthP.resize(frameH * frameW);
    depthX.resize(frameH * frameW);
    depthY.resize(frameH * frameW);

    // Set tube depth for the 2D wave solver
    thrust::fill(depthP.begin(), depthP.end(), 1.0);
    thrust::fill(depthX.begin(), depthX.end(), 1.0);
    thrust::fill(depthY.begin(), depthY.end(), 1.0);

    // Set tube depth for the 2.5D wave solver
    if (fdtd_solver_type == SIMULATION_2_5D){
        TubeDepth tubeDepth(tubeGeometry, depthX, depthY, depthP);
        tubeDepth.setTubeDepth();
    }
   
    // Print depth components [depthP, depthX and depthY]
    // printDepthComponents(thrust::raw_pointer_cast(depthX.data()), "depthX");
    // printDepthComponents(thrust::raw_pointer_cast(depthY.data()), "depthY");
    // printDepthComponents(thrust::raw_pointer_cast(depthP.data()), "depthP");

    cudaFree(boundary_segment_type);
}

// Compute the solver coefficients for the interior grid cells
void FDTDSolver::calculateSolverCoefficients(){

    // Calculate the number of interior cells for which we calculate solver coefficients
    int interior_rows = frameH - 2;
    int interior_cols = frameW - 2;
    int num_interior_cells = interior_rows * interior_cols;

    // Allocate global memory for cell types
    int* d_cur_cell_type;
    int* d_right_cell_type;
    int* d_top_cell_type;
    float* d_cur_beta;
    float* d_right_beta;
    float* d_top_beta;

    cudaMalloc(&d_cur_cell_type, num_interior_cells * sizeof(int));
    cudaMalloc(&d_right_cell_type, num_interior_cells * sizeof(int));
    cudaMalloc(&d_top_cell_type, num_interior_cells * sizeof(int));

    cudaMalloc(&d_cur_beta, num_interior_cells * sizeof(float));
    cudaMalloc(&d_right_beta, num_interior_cells * sizeof(float));
    cudaMalloc(&d_top_beta, num_interior_cells * sizeof(float));

    // Thread configuration
    int NUM_THREADS_PER_BLOCK = 256;
    int NUM_BLOCKS = (num_interior_cells + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

    // Launch kernel to extract the cell types and beta values
    extractCellTypesBetaVals<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(PV_N.data()),
                                                                    num_interior_cells,
                                                                    frameW,
                                                                    d_cur_cell_type, d_right_cell_type, d_top_cell_type,
                                                                    d_cur_beta, d_right_beta, d_top_beta);

    // Print the extracted cell types and beta values to CSV files
    // Print cell_types and veta values to CSV files

    printNeighboringCellsData("cell_type_vals", d_cur_cell_type, d_right_cell_type, d_top_cell_type);
    printNeighboringCellsData("beta_vals", d_cur_beta, d_right_beta, d_top_beta);

    // Allocate global memory for N_out_val and N_in_val

    cudaMalloc(&N_out, num_interior_cells * sizeof(float));
    cudaMalloc(&N_in, num_interior_cells * sizeof(float));

    extractNormalComponents<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(num_interior_cells, N_out, N_in, d_cur_beta, d_right_beta, d_top_beta);

    // Allocate global memory for sigma_prime_dt
    float* d_sigma_prime_dt_cur;
    float* d_sigma_prime_dt_right;
    float* d_sigma_prime_dt_top;

    cudaMalloc(&d_sigma_prime_dt_cur, num_interior_cells * sizeof(float));
    cudaMalloc(&d_sigma_prime_dt_right, num_interior_cells * sizeof(float));
    cudaMalloc(&d_sigma_prime_dt_top, num_interior_cells * sizeof(float));

    extractSigmaPrimedt<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(num_interior_cells, d_cur_cell_type, d_right_cell_type, d_top_cell_type,
                                                               d_sigma_prime_dt_cur, d_sigma_prime_dt_right, d_sigma_prime_dt_top);

    // Print the sigma_prime_dt values to CSV files
    printNeighboringCellsData("sigma_prime_dt_vals", d_sigma_prime_dt_cur, d_sigma_prime_dt_right, d_sigma_prime_dt_top);
    
    // Allocate global memory to verify if the current and neighboring cells are excitation cells or not
    int* is_excitation_cell_cur;
    int* is_excitation_cell_right;
    int* is_excitation_cell_top;

    cudaMalloc(&is_excitation_cell_cur, num_interior_cells * sizeof(int));
    cudaMalloc(&is_excitation_cell_right, num_interior_cells * sizeof(int));
    cudaMalloc(&is_excitation_cell_top, num_interior_cells * sizeof(int));

    isExcitationCell<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(num_interior_cells, d_cur_cell_type, d_right_cell_type, d_top_cell_type,
                                                            is_excitation_cell_cur, is_excitation_cell_right, is_excitation_cell_top);
    
    cudaMalloc(&minVxBeta, num_interior_cells * sizeof(float));
    cudaMalloc(&minVyBeta, num_interior_cells * sizeof(float));

    // Compute minVxBeta and minVyBeta using the extracted beta values
    setMinVxVyBeta<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(num_interior_cells, d_cur_beta, d_right_beta, d_top_beta, minVxBeta, minVyBeta);

    // Find max sigma_prime_dt for Vx and Vy
    // For Vx = max(sigma_prime_dt_cur_cell, sigma_prime_dt_right_cell)
    // For Vy = max(sigma_prime_dt_cur_cell, sigma_prime_dt_top_cell)
    
    cudaMalloc(&maxVxSigmaPrimedt, num_interior_cells * sizeof(float));
    cudaMalloc(&maxVySigmaPrimedt, num_interior_cells * sizeof(float));

    // Set maxVxSigmaPrimedt and maxVySigmaPrimedt using the extracted sigma_prime_dt values
    setMaxVxVySigmaPrimedt<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(num_interior_cells, d_sigma_prime_dt_cur, d_sigma_prime_dt_right, d_sigma_prime_dt_top,
                                                                  maxVxSigmaPrimedt, maxVySigmaPrimedt);  

    // Compute betaVxSqr, betaVySqr, betaVxSqr_dt_inv_rho_inv_ds and betaVySqr_dt_inv_rho_inv_ds

    cudaMalloc(&betaVxSqr, num_interior_cells * sizeof(float));
    cudaMalloc(&betaVySqr, num_interior_cells * sizeof(float));
    cudaMalloc(&betaVxSqr_dt_inv_rho_inv_ds, num_interior_cells * sizeof(float));
    cudaMalloc(&betaVySqr_dt_inv_rho_inv_ds, num_interior_cells * sizeof(float));

    // Compute betaVxSqr and betaVySqr using minVxBeta and minVyBeta
    setBetaVxVyParams<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(num_interior_cells, minVxBeta, minVyBeta, betaVxSqr, betaVySqr, betaVxSqr_dt_inv_rho_inv_ds, betaVySqr_dt_inv_rho_inv_ds);
    
    // Set pressureSigmaPrimedt
    // Don't do a direct copy from d_sigma_prime_dt_cur to pressureSigmaPrimedt
    // as we free the d_sigma_prime_dt_cur global memory later.
    cudaMalloc(&pressureSigmaPrimedt, num_interior_cells * sizeof(float));
    cudaMemcpy(pressureSigmaPrimedt, d_sigma_prime_dt_cur, num_interior_cells * sizeof(float), cudaMemcpyDeviceToDevice);
       
    // Check if the current and neighboring cells are not excitation cells
    // Set areWeNotExcitationVx and areWeNotExcitationVy
 
    cudaMalloc(&areWeNotExcitationVx, num_interior_cells * sizeof(float));
    cudaMalloc(&areWeNotExcitationVy, num_interior_cells * sizeof(float));

    setAreWeNotExcitationVxVy<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(num_interior_cells, is_excitation_cell_cur, is_excitation_cell_right, is_excitation_cell_top, areWeNotExcitationVx, areWeNotExcitationVy);

    // Set exciation weight

    cudaMalloc(&excitationWeightVx, num_interior_cells * sizeof(float));
    cudaMalloc(&excitationWeightVy, num_interior_cells * sizeof(float));

    // Set the excitation weights using is_excitation and source direction
    setExcitationWeightVxVy<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(num_interior_cells, is_excitation_cell_cur, is_excitation_cell_right, is_excitation_cell_top, 
                                                                   excitationWeightVx, excitationWeightVy);
    
    // Set the XOR terms

    cudaMalloc(&xor_val1, num_interior_cells * sizeof(float));
    cudaMalloc(&xor_val2, num_interior_cells * sizeof(float));
    cudaMalloc(&xor_val3, num_interior_cells * sizeof(float));
    cudaMalloc(&xor_val4, num_interior_cells * sizeof(float));

    setXORVals<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(num_interior_cells, d_cur_beta, d_right_beta, d_top_beta,
                                                      xor_val1, xor_val2, xor_val3, xor_val4);
    cudaDeviceSynchronize();
    
    cudaFree(d_cur_cell_type);
    cudaFree(d_right_cell_type);
    cudaFree(d_top_cell_type);
    cudaFree(d_cur_beta);
    cudaFree(d_right_beta);
    cudaFree(d_top_beta);
    cudaFree(d_sigma_prime_dt_cur);
    cudaFree(d_sigma_prime_dt_right);
    cudaFree(d_sigma_prime_dt_top);
    cudaFree(is_excitation_cell_cur);
    cudaFree(is_excitation_cell_right);
    cudaFree(is_excitation_cell_top);
    cudaFree(betaVxSqr);
    cudaFree(betaVySqr);
}

/**************C U D A***K E R N E L***F U N C T I O N S*************/

// Fill the 
__global__
void fillConst(float* data, int N, float val) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    for (; idx < N; idx += stride) {
        data[idx] = val;
    }
}

// CUDA kernel to set domain boundary cells to dead cells
__global__ 
void setupDeadCells(FDTDSolver::GridCellComponents* d_PV_N, int frameH, int frameW, int cell_dead) {

    // Get the thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Update the grid cells types inside the computational domain.
    if (idx <= frameH * frameW) {

        int row = idx / frameW;
        int col = idx % frameW;

        // Define the outermost cells as the dead cells
        // Check if the cell is on the boundary (top, bottom, left, or right)

        if (row == 0 || row == frameH - 1 || col == 0 || col == frameW - 1) {
            d_PV_N[idx].cell_type = cell_dead;
        }
    }
}

// CUDA kernel to set up the PML layers horizontally
__global__ 
void setupHorizontalPMLLayers(FDTDSolver::GridCellComponents* d_PV_N, int cell_pos1, int cell_pos2, int cell_type){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= cell_pos1 && idx <= cell_pos2) {
        d_PV_N[idx].cell_type = cell_type;
    }
}

//CUDA kernel to set up the PML layers vertically
__global__ 
void setupVerticalPMLLayers(FDTDSolver::GridCellComponents* d_PV_N, int frameW, int y_start, int y_end, int col_left, int cell_type){
    
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (idx >= y_start && idx <= y_end) {
            d_PV_N[idx * frameW + col_left].cell_type = cell_type;
        }
}

// CUDA kernel to extract the cell types and beta values
__global__ 
void extractCellTypesBetaVals(FDTDSolver::GridCellComponents* d_PV_N, int num_interior_cells, int frameW, 
                              int* curr_cell_type, int* right_cell_type, int* top_cell_type,
                              float* curr_beta, float* right_beta, float* top_beta) {
    
    // Get the thread index
    // equivalent to (row_idx-1) * (frameW-2) + (col_idx-1)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_interior_cells) {
        int interrior_col = frameW - 2;
        int col_idx = (tid % interrior_col) + 1; // +1 to skip the dead cell layer  
        int row_idx = (tid / interrior_col) + 1; // +1 to skip the dead cell layer

        // Compute the flattened index for each of the referenced cells in PV_N
        int curr_cell_idx = (row_idx * frameW) + col_idx;
        int right_cell_idx = (row_idx * frameW) + (col_idx + 1);
        int top_cell_idx = ((row_idx - 1) * frameW) + col_idx;

        // Extract the cell types for the current, right, and top cells
        curr_cell_type[tid] = d_PV_N[curr_cell_idx].cell_type;
        right_cell_type[tid] = d_PV_N[right_cell_idx].cell_type;
        top_cell_type[tid] = d_PV_N[top_cell_idx].cell_type;

        // Extract the beta values for the current, right, and top cells
        curr_beta[tid] = d_beta[curr_cell_type[tid]];
        right_beta[tid] = d_beta[right_cell_type[tid]];
        top_beta[tid] = d_beta[top_cell_type[tid]];
    }
}

// CUDA kernel to extract the normal components
__global__ 
void extractNormalComponents(int num_interior_cells, float* d_N_out_val, float* d_N_in_val, float* d_cur_beta, float* d_right_beta, float* d_top_beta){

    // Get the thread index
    // equivalent to (row_idx-1) * (frameW-2) + (col_idx-1)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int look_up_idx = 0;

    if (tid < num_interior_cells){ 
        look_up_idx = betaLookUpTable(d_right_beta[tid], d_top_beta[tid]);
        d_N_out_val[tid] = air_normal_component[look_up_idx] * d_cur_beta[tid];
        d_N_in_val[tid] = wall_normal_component[look_up_idx] * (1-d_cur_beta[tid]);

        //printf("look_up_idx = %d, d_cur_beta[%d] = %f, d_right_beta[%d] = %f, d_top_beta[%d] = %f\n", look_up_idx, tid, d_cur_beta[tid], tid, d_right_beta[tid], tid, d_top_beta[tid]);
        //printf("d_N_out_val[%d] = %f, d_N_in_val[%d] = %f\n", tid, d_N_out_val[tid], tid, d_N_in_val[tid]); 
    }
}

// CUDA kernel to extract the sigma_prime_dt values
__global__ 
void extractSigmaPrimedt(int num_interior_cells, int* curr_cell_type, int* right_cell_type, int* top_cell_type,
                         float* d_sigma_prime_dt_cur, float* d_sigma_prime_dt_right, float* d_sigma_prime_dt_top){
    
    // Get the thread index
    // equivalent to (row_idx-1) * (frameW-2) + (col_idx-1)

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_interior_cells){
        d_sigma_prime_dt_cur[tid] = d_sigma_prime_dt[curr_cell_type[tid]];
        d_sigma_prime_dt_right[tid] = d_sigma_prime_dt[right_cell_type[tid]];
        d_sigma_prime_dt_top[tid] = d_sigma_prime_dt[top_cell_type[tid]];  
    }
}

// CUDA kernel to check if the current and neighboring cells are excitation cells
__global__ 
void isExcitationCell(int num_interior_cells, int* curr_cell_type, int* right_cell_type, int* top_cell_type,
                      int* is_excitation_cell_cur, int* is_excitation_cell_right, int* is_excitation_cell_top){

    // Get the thread index
    // equivalent to (row_idx-1) * (frameW-2) + (col_idx-1)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_interior_cells){
        is_excitation_cell_cur[tid] = (curr_cell_type[tid] == cell_excitation) ? 1 : 0;
        is_excitation_cell_right[tid] = (right_cell_type[tid] == cell_excitation) ? 1 : 0;
        is_excitation_cell_top[tid] = (top_cell_type[tid] == cell_excitation) ? 1 : 0;  
    }
}

// CUDA kernel to compute minVxBeta and minVyBeta
__global__
void setMinVxVyBeta(int num_interior_cells, float* d_cur_beta, float* d_right_beta, 
                    float* d_top_beta, float* minVxBeta, float* minVyBeta){

    // Get the thread index
    // equivalent to (row_idx-1) * (frameW-2) + (col_idx-1)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Find the minimum beta values for the current, right, and top cells
    if (tid < num_interior_cells){
        minVxBeta[tid] = min(d_cur_beta[tid], d_right_beta[tid]);
        minVyBeta[tid] = min(d_cur_beta[tid], d_top_beta[tid]);
    }
}

// CUDA kernel to compute maxVxSigmaPrimedt and maxVySigmaPrimedt
__global__ 
void setMaxVxVySigmaPrimedt(int num_interior_cells, float* d_sigma_prime_dt_cur, float* d_sigma_prime_dt_right, 
                            float* d_sigma_prime_dt_top, float* maxVxSigmaPrimedt, float* maxVySigmaPrimedt){

    // Get the thread index
    // equivalent to (row_idx-1) * (frameW-2) + (col_idx-1)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Find the maximum sigma_prime_dt values for the current, right, and top cells
    if (tid < num_interior_cells){
        maxVxSigmaPrimedt[tid] = max(d_sigma_prime_dt_cur[tid], d_sigma_prime_dt_right[tid]);
        maxVySigmaPrimedt[tid] = max(d_sigma_prime_dt_cur[tid], d_sigma_prime_dt_top[tid]);
    }
}

// CUDA kernel to compute betaVxSqr, betaVySqr, betaVxSqr_dt_inv_rho_inv_ds and betaVySqr_dt_inv_rho_inv_ds
__global__ 
void setBetaVxVyParams(int num_interior_cells, float* minVxBeta, float* minVyBeta, float* betaVxSqr, float* betaVySqr, 
                       float* betaVxSqr_dt_inv_rho_inv_ds, float* betaVySqr_dt_inv_rho_inv_ds){    

    // Get the thread index
    // equivalent to (row_idx-1) * (frameW-2) + (col_idx-1)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute betaVxSqr and betaVySqr using minVxBeta and minVyBeta
    if (tid < num_interior_cells){
        betaVxSqr[tid] = minVxBeta[tid] * minVxBeta[tid];
        betaVySqr[tid] = minVyBeta[tid] * minVyBeta[tid];

        betaVxSqr_dt_inv_rho_inv_ds[tid] =(betaVxSqr[tid] * d_dt) / (d_rho* d_ds);
        betaVySqr_dt_inv_rho_inv_ds[tid] =(betaVySqr[tid] * d_dt) / (d_rho* d_ds);
    }
}

// CUDA kernel to set the areWeNotExcitationVx and areWeNotExcitationVy values
__global__ 
void setAreWeNotExcitationVxVy(int num_interior_cells, int* is_excitation_cell_cur, int* is_excitation_cell_right, int* is_excitation_cell_top,
                               float* areWeNotExcitationVx, float* areWeNotExcitationVy){

    // Get the thread index
    // equivalent to (row_idx-1) * (frameW-2) + (col_idx-1)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current and neighboring cells are not excitation cells
    if (tid < num_interior_cells){
        areWeNotExcitationVx[tid] = (1-is_excitation_cell_cur[tid]) * (1-is_excitation_cell_right[tid]);
        areWeNotExcitationVy[tid] = (1-is_excitation_cell_cur[tid]) * (1-is_excitation_cell_top[tid]);
    }
}

// CUDA kernel to set the excitation weights
__global__ 
void setExcitationWeightVxVy(int num_interior_cells, int* is_excitation_cell_cur, int* is_excitation_cell_right, int* is_excitation_cell_top,
                             float* excitationWeightVx, float* excitationWeightVy){

    // Get the thread index
    // equivalent to (row_idx-1) * (frameW-2) + (col_idx-1)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Set the excitation weights using is_excitation and source direction
    if (tid < num_interior_cells){

        excitationWeightVx[tid] = (is_excitation_cell_cur[tid] * d_source_direction[2]) + (is_excitation_cell_right[tid] * d_source_direction[0]);
        excitationWeightVy[tid] = (is_excitation_cell_cur[tid] * d_source_direction[3]) + (is_excitation_cell_top[tid] * d_source_direction[1]);
    }

}

// CUDA kernel to set the XOR values
__global__ 
void setXORVals(int num_interior_cells, float* d_cur_beta, float* d_right_beta, float* d_top_beta,
                float* xor_val1, float* xor_val2, float* xor_val3, float* xor_val4){

    // Get the thread index
    // equivalent to (row_idx-1) * (frameW-2) + (col_idx-1)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Set the XOR values 
    if (tid < num_interior_cells){
        xor_val1[tid] = d_right_beta[tid] *   (1-d_cur_beta[tid]);
        xor_val2[tid] = d_cur_beta[tid]   *   (1-d_right_beta[tid]);
        xor_val3[tid] = d_top_beta[tid]   *   (1-d_cur_beta[tid]);
        xor_val4[tid] = d_cur_beta[tid]   *   (1-d_top_beta[tid]);
    }
}

// CUDA kernel to find the if the neighboring cells are wall cells or air cells
__device__
int betaLookUpTable(int beta_right, int beta_top){

    if      (beta_right == cell_air && beta_top == cell_air) return 0;
    else if (beta_right == cell_air && beta_top == cell_wall) return 1;
    else if (beta_right == cell_wall && beta_top == cell_air) return 2;
    else if (beta_right == cell_wall && beta_top == cell_wall) return 3;
    // For “impossible” combinations, return a sentinel:
    else                                                     return -1;
}

__global__
void setZInverse(float* z_inv, int* boundary_segment_type, 
                 int frameH, int frameW, float z_inv_val){

    // Get the global thread index

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid >= frameH * frameW ) return;

    if (boundary_segment_type[tid] != 0)
        z_inv[tid] = z_inv_val;                
}