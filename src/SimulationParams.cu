#include <cuda_runtime.h>

#include "../include/SimulationParams.h"

// Define the default constructor

SimulationParams::SimulationParams(int srate_mul){

    // Physical constants
    sound_speed = 350.0f * (METER/SECOND);
    rho         = (1.14f * KILOGRAM)/pow(METER, 3);
    base_sample_srate = 44100 * HERTZ;
    kappa = rho * sound_speed * sound_speed; //Bulk modulus
    srate = base_sample_srate * srate_mul;

    //Setup grid resolution
    dt = 1.0f/static_cast<float>(srate);
    ds = dt*sound_speed*sqrtf(2.0f);
                                    
    // Parameters to set up PML layers
    max_sigma_dt = 0.5;
    pml_layers = 6;

    // Vocal tract wall vibration parameters
    wall_pressure_coupling_coeff = 0.0f;
    M0 = 21 * (KILOGRAM/(METER*METER));
    B0 = 8000 * (KILOGRAM/(METER*METER*SECOND));
    K0 = 845000 * (KILOGRAM/(METER*METER*SECOND*SECOND));

    Mw = M0;
    Bw = B0;
    Kw = K0;
}

// Define the constructor to set up user input parameters
SimulationParams::SimulationParams(const int fdtd_solver_type, 
                                   const int simulation_type,
                                   int tube_geometry,
                                   int vowel_type,
                                   int cross_sectional_shape,
                                   const int mouth_radiation_cond,
                                   const int srate_mul, 
                                   const float audio_dur,
                                   const int source_type,
                                   const int pml_flag){
    
    this->fdtd_solver_type = fdtd_solver_type;
    this->simulation_type  = simulation_type;
    this->tube_geometry    = tube_geometry;
    this->vowel_type       = vowel_type;
    this->cross_sectional_shape = cross_sectional_shape;
    this->mouth_radiation_cond = mouth_radiation_cond;
    this->srate_mul        = srate_mul;
    this->audio_dur        = audio_dur;
    this->source_type      = source_type;
    this->pml_flag         = pml_flag;
}


int SimulationParams::getFDTDSolverType(){
    return fdtd_solver_type;
}

float SimulationParams::getAudioDur(){
    return audio_dur;
}

int SimulationParams::getSourceType(){
    return source_type;
}

int SimulationParams::getPMLFlag(){
    return pml_flag;
}

int SimulationParams::getSrateMul(){
    return srate_mul;
}

int   SimulationParams::getSimulationType(){
    return simulation_type;
}

int   SimulationParams::getTubeGeometry(){
    return tube_geometry;
}

int   SimulationParams::getCrossSectionalShape(){
    return cross_sectional_shape;
}

int SimulationParams::getMouthRadiationCond(){
    return mouth_radiation_cond;
}

int SimulationParams::getVowelType(){
    return vowel_type;
}