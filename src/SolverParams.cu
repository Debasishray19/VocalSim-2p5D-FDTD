#include "../include/SolverParams.h"

#include <iostream>

void SolverParams::setBeta(int num_pml_layers){

    // Value of beta can be either 1 or 0
    // For wall cell = 0
    // For air cell = 1
    // For no_pressure cell = 1
    // For excitation cell = 0
    // For head cell = 0
    // For dead cell = 0;

    beta[CellTypes::cell_wall] = 0;
    beta[CellTypes::cell_air] = 1;
    beta[CellTypes::cell_excitation] = 0;
    beta[CellTypes::cell_dynamic] = 0;
    beta[CellTypes::cell_dead] = 0;
    beta[CellTypes::cell_noPressure] = 1;
    beta[CellTypes::cell_head] = 0;

    // Define beta for PML layers
    for (int count=0; count < num_pml_layers; count++){
        beta[CellTypes::cell_pml0+count] = 1;
    }

}

void SolverParams::setSigmaPrimeDt(float max_sigma_dt, float dt, int num_pml_layers){

    // sigma_prime_dt = sigma_prime * dt, where sigma_prime = 1 - beta + sigma]
    // sigma_prime_dt = (1 - beta + sigma) * dt, where sigma = 0 for all non PML layers
    // sigma_prime_dt = (1-beta)*dt

    // For wall cell = (1-0)*dt = dt
    // For air cell = (1-1)*dt = 0
    // For no_pressure cell = (1-1)*dt = 0
    // For excitation cell = (1-0)*dt = dt
    // For head cell = (1-0)*dt = dt
    // For dead cell = 1000000;

    sigma_prime_dt[CellTypes::cell_wall] = dt;
    sigma_prime_dt[CellTypes::cell_air] = 0;
    sigma_prime_dt[CellTypes::cell_noPressure] = 0;
    sigma_prime_dt[CellTypes::cell_excitation] = dt;
    sigma_prime_dt[CellTypes::cell_head] = dt;
    sigma_prime_dt[CellTypes::cell_dynamic] = 0;
    sigma_prime_dt[CellTypes::cell_dead] = 1000000;

    // Define sigma_prime_dt for PML layers
    // For PML layers beta = 1
    // sigma_prime_dt = sigma * dt
    for(int count=0; count < num_pml_layers; count++){
        sigma_prime_dt[CellTypes::cell_pml0+count] = ((float)count/(float)(num_pml_layers-1)) * max_sigma_dt;
    }
}

void SolverParams::setTimeStep(float audio_time, float dt){
    float init_t = 0;

    while (init_t <= audio_time-dt){
        time_step.push_back(init_t);
        init_t = init_t + dt;
    }
}

void SolverParams::setSourceDirection(){
    
    // Define source propagation direction

    source_direction[0] = 0;  // Left  = -1
    source_direction[1] = 0;  // Down  = -1
    source_direction[2] = 1;  // Right = 1
    source_direction[3] = 0;  // Up    = 1
}

float* SolverParams::getBeta(){
    return beta;
}

float* SolverParams::getSigmaPrimeDt(){
    return sigma_prime_dt;
}

std::vector<float> SolverParams::getTimeStep(){
    return time_step;
}

float* SolverParams::getSourceDirection(){
    return source_direction;
}