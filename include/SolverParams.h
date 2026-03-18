/*
 * SolverParam.h
 * 
 * Define solver parameters
 * 
 */

#ifndef SOLVERPARAMS_H_
#define SOLVERPARAMS_H_

#include <vector>
#include <cmath>

#include "SimulationParams.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define NUM_CELL_TYPES  13

// Define grid cell types
enum CellTypes{
    cell_wall,          //0
    cell_air,           //1
    cell_excitation,    //2
    cell_pml0,          //3
    cell_pml1,          //4
    cell_pml2,          //5
    cell_pml3,          //6
    cell_pml4,          //7
    cell_pml5,          //8
    cell_dynamic,       //9 [currently, I am not using cell_dynamic]
    cell_dead,          //10
    cell_noPressure,    //11
    cell_head,          //12
};

class SolverParams{

    private:

        std::vector<float> time_step;
        float source_direction[4];
        float beta[NUM_CELL_TYPES];
        float sigma_prime_dt [NUM_CELL_TYPES];

    public:

        void setBeta(int num_pml_layers);
        void setSigmaPrimeDt(float max_sigma_dt, float dt, int num_pml_layers);
        void setTimeStep(float audio_dur, float dt);
        void setSourceDirection();

        float* getBeta();
        float* getSigmaPrimeDt();
        std::vector<float> getTimeStep();
        float* getSourceDirection();
        
};
 
#endif // SOLVERPARAMS_H_