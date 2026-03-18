#include "../include/GlottalExcitations.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Set the excitation velocity for various simulation types

void GlottalExcitations::setExcitationVelocity(int source_type, int num_steps, double dt){
    
    // Initiate the excitation velocity as zero
    host_excitation_velocity.resize(num_steps);
    thrust::fill(host_excitation_velocity.begin(), host_excitation_velocity.end(), 0.0);

    switch(source_type) {
        case 0:{ // sinusoidal excitation

            // Define source parameters
            float source_freq = 440 * HERTZ;
            float max_source_amplitude = 25;

            for (int idx = 0; idx < num_steps; idx++) {
                float t = idx * dt;
                host_excitation_velocity[idx] = max_source_amplitude * sin(2 * M_PI * source_freq * t);
            }
        } break;

        case 1:{ // Gaussian excitation
            float f0 = 10.0f * KILOHERTZ;
            float bell_peak_pos = 0.646f/f0;
            float bell_width = 0.29f * bell_peak_pos;
            float amp_scaling_factor = 0.025f;

            for (int idx = 0; idx < num_steps; idx++){
                float t = idx * dt;
                float exponent_val = (t-bell_peak_pos)/bell_width;
                float sqr_exponent_val = exponent_val * exponent_val;
                host_excitation_velocity[idx] = amp_scaling_factor * expf(-1 * sqr_exponent_val);
            }
        } break;

        case 2:{ // Vocal fold model - Two mass model
            TwoMassModel twoMassModel;
            twoMassModel.setTwoMassModelParams();
        }break;

        default:{
            std::cerr << "Invalid source type" << std::endl;
            exit(EXIT_FAILURE);
        } break;
    }
}


// Return the excitation velocity as a host vector

thrust::host_vector<float> GlottalExcitations::getExcitationVelocity(){
    return host_excitation_velocity;
}