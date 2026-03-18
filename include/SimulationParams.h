/*
 * SimulationParams.h
 * 
 * Define simulation parameters
 * 
 */

#ifndef SIMULATIONPARAMS_H_
#define SIMULATIONPARAMS_H_
 
#include "SimulationUnits.h"
#include <cmath>
#include <iostream>
#include<stdio.h>

// Check error codes for CUDA functions
#define CHECK(call){                                           \
    cudaError_t error = call;                                  \
    if (error != cudaSuccess){                                 \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error,       \
        cudaGetErrorString(error));                            \
    }                                                          \
}
 
class SimulationParams{
    
    private:

        // Simulation setup parameters - Input by user

        int fdtd_solver_type;
        int simulation_type;
        int tube_geometry;
        int vowel_type;
        int cross_sectional_shape;
        int mouth_radiation_cond;
        int srate_mul;
        float audio_dur;
        int source_type;
        int pml_flag;
    
    public:

        // Physical constants

        float sound_speed; 
        float rho;
        int base_sample_srate;
        float kappa; //Bulk modulus
        int srate;

        // Simulation Constants for PML layers

        float max_sigma_dt;
        int pml_layers;
 
        // Vocal tract wall parameters
        float wall_pressure_coupling_coeff;
        float M0;
        float B0;
        float K0;
 
        float Mw;
        float Bw;
        float Kw;

        // Set up the grid resolution
        float ds; // Spatial resolution
        float dt; // Temporal resolution
     
    public:

        SimulationParams(int srate_mul);

        SimulationParams(const int fdtd_solver_type, 
                         const int simulation_type,
                         int tube_geometry,
                         int vowel_type,
                         int cross_sectional_shape,
                         const int mouth_radiation_cond,
                         const int srate_mul, 
                         const float audio_dur,
                         const int source_type,
                         const int pml_flag);
    
    public: 

        int   getFDTDSolverType();
        int   getSimulationType();
        int   getTubeGeometry();
        int   getVowelType();
        int   getCrossSectionalShape();
        int   getMouthRadiationCond();
        int   getSrateMul();
        float getAudioDur();
        int   getSourceType();
        int   getPMLFlag();
     
};
 
#endif // SIMULATIONPARAMS_H_