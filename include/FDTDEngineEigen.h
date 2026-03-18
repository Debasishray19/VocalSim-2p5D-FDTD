/*
 * FDTDEngineEigen.h
 * 
 * Vectorization of FDTDEngineCPU using the Eigen library
 * 
 */

#ifndef FDTDENGINEEIGEN_H_
#define FDTDENGINEEIGEN_H_

#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <thrust/host_vector.h>

#include "FDTDSolver.h"
#include "FDTDEngineCPU.h"
#include "TwoMassModel.h"

using namespace std;

using MatrixXf    = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXi    = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXf = Eigen::Matrix<float, Eigen::Dynamic, 1>;

class FDTDEngineEigen{

    public:

        int frameH;
        int frameW;
        int listenerX;
        int listenerY;

        // Declare solver params
        MatrixXf CxVx;
        MatrixXf CyVy;
        MatrixXf CxP;
        MatrixXf CyP;
        MatrixXf Vb_alphaX;
        MatrixXf Vb_alphaY;

        // Declare solver coefficients
        MatrixXf N_out_mat;
        MatrixXf N_in_mat;
        MatrixXf minVxBeta_mat;
        MatrixXf minVyBeta_mat;
        MatrixXf maxVxSigmaPrimedt_mat;
        MatrixXf maxVySigmaPrimedt_mat;
        MatrixXf betaVxSqr_dt_inv_rho_inv_ds_mat;
        MatrixXf betaVySqr_dt_inv_rho_inv_ds_mat;
        MatrixXf pressureSigmaPrimedt_mat;
        MatrixXf areWeNotExcitationVx_mat;
        MatrixXf areWeNotExcitationVy_mat;
        MatrixXf excitationWeightVx_mat;
        MatrixXf excitationWeightVy_mat;
        MatrixXf xor_val1_mat;
        MatrixXf xor_val2_mat;
        MatrixXf xor_val3_mat;
        MatrixXf xor_val4_mat;
        MatrixXf z_inv_mat;

        MatrixXf w_prev; // Rate of change of wall displacement
        MatrixXf w_next;
        MatrixXf z_prev; // Wall displacement
        MatrixXf z_next;

        // Declare depth components
        MatrixXf depthP;
        MatrixXf depthX;
        MatrixXf depthY;

        // Grid cell params - Instead of declaring as a single structure
        // split the acoustic components [Pr, Vx, Vy and cell_types] for 
        // code readability and to avoid compile-time errors
        MatrixXf Pr_N;
        MatrixXf Pr_Nplus1;
        MatrixXf Vx_N;
        MatrixXf Vx_Nplus1;
        MatrixXf Vy_N;
        MatrixXf Vy_Nplus1;

        MatrixXi cell_type;

        VectorXf record_pressure; // Record pressure samples at listener position

    private:

        FDTDSolver& _fdtdSolver;  // <- To setup the computational domain and the tube geometry 
        FDTDEngineCPU& _fdtdEngineCPU; // <- To transfer the variables from device to host memory

    public:

        FDTDEngineEigen(FDTDEngineCPU& fdtdEngineCPU, FDTDSolver& fdtdSolver, bool validationFlag = false);

        void startSolverEngine();
        
    private:

        void update_wall_pres_and_wall_disp(int t_step);
        void update_velocity(float exeV);
        void update_border_cells();

        void initializeSolverCoeffs();
        void assignSolverCoeffs();
        void printMatrixInCSV(const string& file_name, const MatrixXf& data_mat);
        void printMatrixInCSV(const string& file_name, const MatrixXi& data_mat);
        void validateSolverCoefficients();
        void savePressureSamples();
        void printDepthComponents();
        void printGridCellComponents();

};

#endif // FDTDENGINEEIGEN_H_