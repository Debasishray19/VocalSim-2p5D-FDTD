#include "TwoMassModel.h"
#include "SolverParams.h"

#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>

void testTwoMassModel(){

    TwoMassModel twoMassModel;
    twoMassModel.setTwoMassModelParams();

    // Set sample rate multiplier
    int srate_mul    = 15;
    float audio_time = 1.0f; // 1 second of audio
    float dt         = 1.0f/static_cast<float>(44100 * srate_mul); // Assuming a sample rate of 44100 Hz

    SolverParams solverParams;
    solverParams.setTimeStep(audio_time, dt);

    vector<float> time_step = solverParams.getTimeStep();
    int num_steps = time_step.size();

    vector<float> output_ug(num_steps, 0.0f);

    for(int i = 0; i < num_steps; i++){
        twoMassModel.runTwoMassModel(srate_mul);
        output_ug[i] = twoMassModel.getVolumeVelocity();
    }

    // Find the max volume velocity
    float max_ug = *std::max_element(output_ug.begin(), output_ug.end());

    // Save the normalized volume velocity to a file
    std::ofstream out("../../results/norm_ug.txt");
    if(!out){
        std::cerr << "ERROR: could not open output file.\n";
        return;
    }

    // Write the normalized volume velocity to the file
    for(int i = 0; i < num_steps; ++i){
        float t_milli = time_step[i] * 1000.0f;
        float norm_ug = output_ug[i] / max_ug;
        out << t_milli << '\t' << norm_ug << '\n';
    }

    out.close();

    std::cout << "Completed writing " << num_steps << " samples to norm_ug.txt\n";

    // Finish the simulation
    exit(EXIT_SUCCESS);
}