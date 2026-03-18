/*
 * FDTDSolver.h
 * 
 * SetUp finite-different time-domain numerical wave solver
 * 
 */

#ifndef FDTDSOLVER_H_
#define FDTDSOLVER_H_

#include "SimulationUnits.h"
#include "SimulationParams.h"
#include "SolverParams.h"
#include "GlottalExcitations.h"
#include "TubeDepth.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define SIMULATION_2D 0
#define SIMULATION_2_5D 1 

using namespace std;

// forward‐declare TubeGeometry 
class TubeGeometry;

class FDTDSolver{

    public:

        int fdtd_solver_type;
        int simulation_type;
        int tube_geometry;
        int vowel_type;
        int cross_sectional_shape;
        int mouth_radiation_condition;
        float audio_dur;
        int source_type;
        int pml_flag;
        int srate_mul;

        float kappa;
        float rho_sqrC_dt_inv_ds;
        float rho_sqrC_dt;
        float open_space_depth;
        int num_pml_layers;

        float sound_speed;
        float rho;
        float wall_pressure_coupling_coeff;
        float Mw;
        float Bw;
        float Kw;
        float ds;
        float dt;

        int frameH;
        int frameW;
        int domainH;
        int domainW;

        int listenerY;
        int listenerX;
        int excitationY;
        int excitationX;

        float max_sigma_dt;
        float* beta;
        float* sigma_prime_dt;
        float* source_direction; 

        vector<float> time_step;
        int num_steps;
    
        struct GridCellComponents{
            float Pr;       // Pressure
            float Vx;       // Velocity along X
            float Vy;       // Velocity along Y
            int cell_type;  // cell_type
        };

        thrust::device_vector<GridCellComponents> PV_N;
        thrust::device_vector<GridCellComponents> PV_Nplus1;

        // This one is required to update the velocity components and avoid race conditions
        thrust::device_vector<GridCellComponents> PV_Nplus1_out; 

        thrust::device_vector<float> depthP;
        thrust::device_vector<float> depthX;
        thrust::device_vector<float> depthY;

        thrust::device_vector<float> excitation_velocity;
    
    private:

        float* N_out = nullptr;
        float* N_in = nullptr;
        float* minVxBeta = nullptr;
        float* minVyBeta = nullptr;
        float* maxVxSigmaPrimedt = nullptr;
        float* maxVySigmaPrimedt = nullptr;
        float* betaVxSqr = nullptr;
        float* betaVySqr = nullptr;
        float* betaVxSqr_dt_inv_rho_inv_ds = nullptr;
        float* betaVySqr_dt_inv_rho_inv_ds = nullptr;
        float* pressureSigmaPrimedt = nullptr;
        float* areWeNotExcitationVx = nullptr;
        float* areWeNotExcitationVy = nullptr;
        float* excitationWeightVx = nullptr;
        float* excitationWeightVy = nullptr;
        float* xor_val1 = nullptr;
        float* xor_val2 = nullptr;
        float* xor_val3 = nullptr;
        float* xor_val4 = nullptr;
        int* boundary_segment_type = nullptr;
        float* z_inv = nullptr;

    public:

        FDTDSolver(SimulationParams simulationParams);
   
    public:

        // Set up the computational wave solver
        void setupWaveSolver();

        // For debugging purposes - Print various parameters
        void printInMatrixFormat(float* data, string file_name, int num_rows, int num_cols);
        void printInMatrixFormat(int* data, string file_name, int num_rows, int num_cols);
        void printGridCellComponents();
        void printNeighboringCellsData(string file_name, float* data_cur, float* data_right, float* data_top);
        void printNeighboringCellsData(string file_name, int* data_cur, int* data_right, int* data_top);
        void printDepthComponents(float* data, string file_name);

        // Set getter functions to access solver coefficients
        float* getN_out() const { return N_out; }
        float* getN_in() const { return N_in; }
        float* getMinVxBeta() const { return minVxBeta; }
        float* getMinVyBeta() const { return minVyBeta; }
        float* getMaxVxSigmaPrimedt() const { return maxVxSigmaPrimedt; }
        float* getMaxVySigmaPrimedt() const { return maxVySigmaPrimedt; }
        float* getBetaVxSqr_dt_inv_rho_inv_ds() const { return betaVxSqr_dt_inv_rho_inv_ds; }
        float* getBetaVySqr_dt_inv_rho_inv_ds() const { return betaVySqr_dt_inv_rho_inv_ds; }
        float* getPressureSigmaPrimedt() const { return pressureSigmaPrimedt; }
        float* getAreWeNotExcitationVx() const { return areWeNotExcitationVx; }
        float* getAreWeNotExcitationVy() const { return areWeNotExcitationVy; }
        float* getExcitationWeightVx() const { return excitationWeightVx; }
        float* getExcitationWeightVy() const { return excitationWeightVy; }
        float* getXorVal1() const { return xor_val1; }
        float* getXorVal2() const { return xor_val2; }
        float* getXorVal3() const { return xor_val3; }
        float* getXorVal4() const { return xor_val4; }
        int* getBoundarySegmentType(){return boundary_segment_type;}
        float* getZ_inv() const { return z_inv; }

        thrust::device_vector<float>&  getDepthP()           { return depthP; }
        thrust::device_vector<float>&  getDepthX()           { return depthX; }
        thrust::device_vector<float>&  getDepthY()           { return depthY; }
        thrust::device_vector<GridCellComponents>& getPV_N() { return PV_N;   }
        thrust::device_vector<GridCellComponents>& getPV_Nplus1() { return PV_Nplus1; }
        thrust::device_vector<GridCellComponents>& getPV_Nplus1_out() { return PV_Nplus1_out; }
        thrust::device_vector<float>& getExcitationVelocity() { return excitation_velocity; }

    private:
        
        void openSpaceSimulation();
        void tubeGeometrySimulation();
        void calculateSolverCoefficients();
};

#endif // FDTDSOLVER_H_