// STEP1: Clean the build folder [if exist], 
// cmake .. -G "Visual Studio 17 2022" -A x64
// cmake --build . --config Release --verbose

// Before pushing to git: git rm -r --cached build/

#include <iostream>
#include <cuda_runtime.h>

#include "../include/SimulationUnits.h"
#include "../include/SimulationParams.h"
#include "../include/FDTDSolver.h"
#include "../include/FDTDEngineCPU.h"
#include "../include/FDTDEngineEigen.h"
#include "../include/TestTwoMassModel.h"

#define CPU_SEQUENTIAL 1
#define CPU_EIGEN 2

using namespace std;

/*********PRINT GPU SPECS***********/

void printGPUSpecs(){

    // Open the file "data/device_specs.txt" in write mode.
    FILE *fp = fopen("../../data/device_specs.txt", "w");
    if (fp == NULL) {
        // Print an error message and return, if the file could not be opened.
        fprintf(stderr, "Error: failed to open file 'data/device_specs.txt' for writing.\n");
        return;
    }

    int nDevices;
    cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        fprintf(fp, "Device Number: %d\n", i);
        fprintf(fp, "Device name: %s\n", deviceProp.name);
        fprintf(fp, "Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        fprintf(fp, "Total global memory: %zu GB\n", deviceProp.totalGlobalMem / 1024 / 1024 / 1000);
        fprintf(fp, "Total constant memory: %zu KB\n", deviceProp.totalConstMem / 1024);
        fprintf(fp, "Total shared memory per thread block: %zu KB\n", deviceProp.sharedMemPerBlock / 1024);
        fprintf(fp, "Number of streaming multiprocessors GPU has: %d\n", deviceProp.multiProcessorCount);
        fprintf(fp, "Total threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        fprintf(fp, "Total registers per block: %d\n", deviceProp.regsPerBlock);
        fprintf(fp, "Warp size: %d\n", deviceProp.warpSize);
        fprintf(fp, "Max dims of a block: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        fprintf(fp, "Max dims of a grid: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        fprintf(fp, "Memory Clock Rate (KHz): %d\n", deviceProp.memoryClockRate);
        fprintf(fp, "Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
        fprintf(fp, "Peak Memory Bandwidth: %f GB/s\n", 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6);
        fprintf(fp, "Does my device support concurrent copy and execution?: %d\n", deviceProp.asyncEngineCount);
    }

    // Close the file to ensure the data is written.
    fclose(fp);
}

/***********************************/

int main(){

    // Print GPU specs
	const int GPU_SPECS = 1;
    if(GPU_SPECS)
        printGPUSpecs();

    cout << "======================================" <<endl;
    
	const int FDTD_SOLVER_TYPE = 1;
	cout << "FDTD solver type [0: 2D Solver 1: 2.5D Solver] = " << FDTD_SOLVER_TYPE << endl;

    const int SRATE_MUL = 15;
	cout << "Sampling rate multiplier = " << SRATE_MUL << endl;

    const int SIMULATION_TYPE = 1;
    cout << "Simulation type [0: Open space 1: Tube geometry] = " << SIMULATION_TYPE << endl;

    int TUBE_GEOMETRY = 1;
    cout << "Tube geometry [0: Regular 1: Vocal tract] = " << TUBE_GEOMETRY << endl;

    int VOWEL_TYPE = 1;
    cout << "Vowel type [1: /a/ 2: /i/ 3: /u/] = " << VOWEL_TYPE << endl;

    int CROSS_SECTIONAL_SHAPE = 1;
    cout << "Tube cross-sectional shape [1: Circular 2: Elliptical 3: Single-plane symmetry] = " << CROSS_SECTIONAL_SHAPE << endl;

    const int MOUTH_RADIATION = 1;
    cout << "Mouth-end radiation condition [0: No-Dirichlet 1: Dirichlet] = " << MOUTH_RADIATION << endl;

	const float AUDIO_DUR = 100 * MILLISECOND;
	cout << "Simulation audio duration [in second] = " << AUDIO_DUR << endl;

	const int SOURCE_TYPE = 1;
	cout << "Source type [0: Sine wave 1: Gaussian 2: Vocal fold model] = " << SOURCE_TYPE << endl;

	const int PML_FLAG = 0;
	cout << "PML flag [0: OFF 1: ON] = " << PML_FLAG << endl;

    cout << "======================================" <<endl;

    // Test the vocal fold model
    // testTwoMassModel();

    // Ensure TUBE_GEOMETRY and CROSS_SECTIONAL_SHAPE parameters set to -1
    // for open space simulation
    if (SIMULATION_TYPE == 0){
        TUBE_GEOMETRY = -1;
        CROSS_SECTIONAL_SHAPE = -1;
    }

    // Set up simulation parameters - input by user
    SimulationParams simulationParams(FDTD_SOLVER_TYPE, SIMULATION_TYPE, 
                                      TUBE_GEOMETRY, VOWEL_TYPE, CROSS_SECTIONAL_SHAPE,
                                      MOUTH_RADIATION, SRATE_MUL, AUDIO_DUR, 
                                      SOURCE_TYPE, PML_FLAG);
    
    // Set up computational domain for the FDTD solver using simulation parameters
    FDTDSolver fdtdSolver(simulationParams);
    fdtdSolver.setupWaveSolver();

    int solver_engine_type = CPU_EIGEN;

    if (solver_engine_type == CPU_SEQUENTIAL){

        // Transfer the variables from device to host memory
        // Represent these variables as 2D vector containers
        FDTDEngineCPU fdtdEngineCPU(fdtdSolver, false);
        
        // Start the solver engine
        fdtdEngineCPU.startSolverEngine();

    }else if (solver_engine_type == CPU_EIGEN){

        // Transfer the variables from device to host memory
        // Represent these variables as 2D vector containers
        FDTDEngineCPU fdtdEngineCPU(fdtdSolver, false);
        FDTDEngineEigen fdtdEngineEigen(fdtdEngineCPU, fdtdSolver, false);

        // Start the solver engine
        fdtdEngineEigen.startSolverEngine();
    }

    return EXIT_SUCCESS;
}