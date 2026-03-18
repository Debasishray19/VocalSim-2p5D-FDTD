/*
 * FDTDEngineCPU.h
 * 
 * 2D FDTD engine in CPU for simulating sound propagation in a 2D domain.
 * 
 */

#ifndef FDTDENGINECPU_H_
#define FDTDENGINECPU_H_

#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <chrono>


#include "FDTDSolver.h"
#include "DeviceSymbols.h"
#include "TwoMassModel.h"

class FDTDEngineCPU{

    public:

        int frameH;
        int frameW;

        int listenerX;
        int listenerY;

        vector<vector<FDTDSolver::GridCellComponents>> PV_N;
        vector<vector<FDTDSolver::GridCellComponents>> PV_Nplus1;

        vector<vector<float>> depthP;
        vector<vector<float>> depthX;
        vector<vector<float>> depthY;

        vector<vector<float>> CxVx;
        vector<vector<float>> CyVy;
        vector<vector<float>> CxP;
        vector<vector<float>> CyP;
        vector<vector<float>> Vb_alphaX;
        vector<vector<float>> Vb_alphaY;

        vector<vector<float>> N_out_mat;
        vector<vector<float>> N_in_mat;
        vector<vector<float>> minVxBeta_mat;
        vector<vector<float>> minVyBeta_mat;
        vector<vector<float>> maxVxSigmaPrimedt_mat;
        vector<vector<float>> maxVySigmaPrimedt_mat;
        vector<vector<float>> betaVxSqr_dt_inv_rho_inv_ds_mat;
        vector<vector<float>> betaVySqr_dt_inv_rho_inv_ds_mat;
        vector<vector<float>> pressureSigmaPrimedt_mat;
        vector<vector<float>> areWeNotExcitationVx_mat;
        vector<vector<float>> areWeNotExcitationVy_mat;
        vector<vector<float>> excitationWeightVx_mat;
        vector<vector<float>> excitationWeightVy_mat;
        vector<vector<float>> xor_val1_mat;
        vector<vector<float>> xor_val2_mat;
        vector<vector<float>> xor_val3_mat;
        vector<vector<float>> xor_val4_mat;
        vector<vector<float>> z_inv_mat;

        vector<vector<float>> w_prev; // Rate of change of wall displacement
        vector<vector<float>> w_next;
        vector<vector<float>> z_prev; // Wall displacement
        vector<vector<float>> z_next;

        vector<float> record_pressure; // Record pressure samples at listener position

    private:

        float* __restrict__ N_out                       = nullptr;
        float* __restrict__ N_in                        = nullptr;
        float* __restrict__ minVxBeta                   = nullptr;
        float* __restrict__ minVyBeta                   = nullptr;
        float* __restrict__ maxVxSigmaPrimedt           = nullptr;
        float* __restrict__ maxVySigmaPrimedt           = nullptr;
        float* __restrict__ betaVxSqr_dt_inv_rho_inv_ds = nullptr;
        float* __restrict__ betaVySqr_dt_inv_rho_inv_ds = nullptr;
        float* __restrict__ pressureSigmaPrimedt        = nullptr;
        float* __restrict__ areWeNotExcitationVx        = nullptr;
        float* __restrict__ areWeNotExcitationVy        = nullptr;
        float* __restrict__ excitationWeightVx          = nullptr;
        float* __restrict__ excitationWeightVy          = nullptr;
        float* __restrict__ xor_val1                    = nullptr;
        float* __restrict__ xor_val2                    = nullptr;
        float* __restrict__ xor_val3                    = nullptr;
        float* __restrict__ xor_val4                    = nullptr;
        float* __restrict__ z_inv                       = nullptr;
        
        thrust::host_vector<float> h_N_out;
        thrust::host_vector<float> h_N_in;
        thrust::host_vector<float> h_minVxBeta;
        thrust::host_vector<float> h_minVyBeta;
        thrust::host_vector<float> h_maxVxSigmaPrimedt;
        thrust::host_vector<float> h_maxVySigmaPrimedt;
        thrust::host_vector<float> h_betaVxSqr_dt_inv_rho_inv_ds;
        thrust::host_vector<float> h_betaVySqr_dt_inv_rho_inv_ds;
        thrust::host_vector<float> h_pressureSigmaPrimedt;
        thrust::host_vector<float> h_areWeNotExcitationVx;
        thrust::host_vector<float> h_areWeNotExcitationVy;
        thrust::host_vector<float> h_excitationWeightVx;
        thrust::host_vector<float> h_excitationWeightVy;
        thrust::host_vector<float> h_xor_val1;
        thrust::host_vector<float> h_xor_val2;
        thrust::host_vector<float> h_xor_val3;
        thrust::host_vector<float> h_xor_val4;
        thrust::host_vector<float> h_z_inv;
    
        thrust::host_vector<FDTDSolver::GridCellComponents> h_PV_N;
        thrust::host_vector<FDTDSolver::GridCellComponents> h_PV_Nplus1;

        thrust::host_vector<float> h_depthP;
        thrust::host_vector<float> h_depthX;
        thrust::host_vector<float> h_depthY;

        FDTDSolver& _fdtdSolver;
        
    public:

        explicit FDTDEngineCPU(FDTDSolver& fdtdSolver, bool validationFlag = false);
        
    public:

        void startSolverEngine();
        
    private:

        void validateSolverCoefficients();
        void freeCUDAMemory();
        void copySolverParamsFromDeviceToHost();
        void solverParamsInMat();
        void savePressureSamples();

        void printMatrixInCSV(vector<vector<float>> dataToPrint, string file_name, int num_cols, int num_rows);
        void printGridCellComponents();
        void printDepthComponents();
        
        void update_wall_pres_and_wall_disp(int t_step);
        void update_velocity(float exeV);
        void update_border_cells();        

};
#endif // FDTDENGINECPU_H_