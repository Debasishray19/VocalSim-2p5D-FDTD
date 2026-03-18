/*
 * GlottalExcitations.h
 * 
 * Define glottal excitation functions
 * 
 */

#ifndef GLOTTALEXCITATIONS_H_
#define GLOTTALEXCITATIONS_H_
  
#include <cmath>
#include <thrust/host_vector.h>

#include "SimulationUnits.h"
#include "TwoMassModel.h"
  
#define _USE_MATH_DEFINES
  
class GlottalExcitations{
  
    private:
        thrust::host_vector<float> host_excitation_velocity;
     
    public:
        void setExcitationVelocity(int source_type, int num_steps, double dt);
        thrust::host_vector<float> getExcitationVelocity();
};
  
#endif // GLOTTALEXCITATIONS_H_