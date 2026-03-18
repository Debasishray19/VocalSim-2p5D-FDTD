#include "../include/FDTDEngineCPU.h"

// Initialize the solver coefficients using the solver engine
FDTDEngineCPU::FDTDEngineCPU(FDTDSolver& fdtdSolver, bool validationFlag):_fdtdSolver(fdtdSolver){

    // Computational domain size
    this -> frameH       = _fdtdSolver.frameH;
    this -> frameW       = _fdtdSolver.frameW;
   
    // Find the listener position
    this -> listenerX    = _fdtdSolver.listenerX;
    this -> listenerY    = _fdtdSolver.listenerY;

    this -> N_out                       = _fdtdSolver.getN_out();
    this -> N_in                        = _fdtdSolver.getN_in();
    this -> minVxBeta                   = _fdtdSolver.getMinVxBeta();
    this -> minVyBeta                   = _fdtdSolver.getMinVyBeta();
    this -> maxVxSigmaPrimedt           = _fdtdSolver.getMaxVxSigmaPrimedt();
    this -> maxVySigmaPrimedt           = _fdtdSolver.getMaxVySigmaPrimedt();
    this -> betaVxSqr_dt_inv_rho_inv_ds = _fdtdSolver.getBetaVxSqr_dt_inv_rho_inv_ds();
    this -> betaVySqr_dt_inv_rho_inv_ds = _fdtdSolver.getBetaVySqr_dt_inv_rho_inv_ds();
    this -> pressureSigmaPrimedt        = _fdtdSolver.getPressureSigmaPrimedt();
    this -> areWeNotExcitationVx        = _fdtdSolver.getAreWeNotExcitationVx();
    this -> areWeNotExcitationVy        = _fdtdSolver.getAreWeNotExcitationVy();
    this -> excitationWeightVx          = _fdtdSolver.getExcitationWeightVx();
    this -> excitationWeightVy          = _fdtdSolver.getExcitationWeightVy();
    this -> xor_val1                    = _fdtdSolver.getXorVal1();
    this -> xor_val2                    = _fdtdSolver.getXorVal2();
    this -> xor_val3                    = _fdtdSolver.getXorVal3();
    this -> xor_val4                    = _fdtdSolver.getXorVal4();
    this -> z_inv                       = _fdtdSolver.getZ_inv();

    // Define number number of cells for the solver coefficients
    size_t rows = frameH - 2;
    size_t cols = frameW - 2;
    int num_cells = rows * cols;

    h_N_out.resize(num_cells);
    h_N_in.resize(num_cells);
    h_minVxBeta.resize(num_cells);
    h_minVyBeta.resize(num_cells);
    h_maxVxSigmaPrimedt.resize(num_cells);
    h_maxVySigmaPrimedt.resize(num_cells);
    h_betaVxSqr_dt_inv_rho_inv_ds.resize(num_cells);
    h_betaVySqr_dt_inv_rho_inv_ds.resize(num_cells);
    h_pressureSigmaPrimedt.resize(num_cells);
    h_areWeNotExcitationVx.resize(num_cells);
    h_areWeNotExcitationVy.resize(num_cells);
    h_excitationWeightVx.resize(num_cells);
    h_excitationWeightVy.resize(num_cells);
    h_xor_val1.resize(num_cells);
    h_xor_val2.resize(num_cells);
    h_xor_val3.resize(num_cells);
    h_xor_val4.resize(num_cells);
    h_z_inv.resize(frameH * frameW);

    h_PV_N.resize(_fdtdSolver.PV_N.size());
    h_PV_Nplus1.resize(_fdtdSolver.PV_Nplus1.size());

    h_depthP.resize(_fdtdSolver.depthP.size());
    h_depthX.resize(_fdtdSolver.depthX.size());
    h_depthY.resize(_fdtdSolver.depthY.size());

    // Copy data from device to host
    copySolverParamsFromDeviceToHost();

    // Resize vectors to represent solver coefficients in matrix format
    
    N_out_mat                         .resize(rows, vector<float>(cols));
    N_in_mat                          .resize(rows, vector<float>(cols));
    minVxBeta_mat                     .resize(rows, vector<float>(cols));
    minVyBeta_mat                     .resize(rows, vector<float>(cols));
    maxVxSigmaPrimedt_mat             .resize(rows, vector<float>(cols));
    maxVySigmaPrimedt_mat             .resize(rows, vector<float>(cols));
    betaVxSqr_dt_inv_rho_inv_ds_mat   .resize(rows, vector<float>(cols));
    betaVySqr_dt_inv_rho_inv_ds_mat   .resize(rows, vector<float>(cols));
    pressureSigmaPrimedt_mat          .resize(rows, vector<float>(cols));
    areWeNotExcitationVx_mat          .resize(rows, vector<float>(cols));
    areWeNotExcitationVy_mat          .resize(rows, vector<float>(cols));
    excitationWeightVx_mat            .resize(rows, vector<float>(cols));
    excitationWeightVy_mat            .resize(rows, vector<float>(cols));
    xor_val1_mat                      .resize(rows, vector<float>(cols));
    xor_val2_mat                      .resize(rows, vector<float>(cols));
    xor_val3_mat                      .resize(rows, vector<float>(cols));
    xor_val4_mat                      .resize(rows, vector<float>(cols));
    z_inv_mat                         .resize(frameH, vector<float>(frameW));

    PV_N     .resize(frameH, vector<FDTDSolver::GridCellComponents>(frameW));
    PV_Nplus1.resize(frameH, vector<FDTDSolver::GridCellComponents>(frameW));
    depthP   .resize(frameH, vector<float>(frameW));
    depthX   .resize(frameH, vector<float>(frameW));
    depthY   .resize(frameH, vector<float>(frameW));

    // Matrix representation of solver coefficients
    solverParamsInMat();

    // Validate the solver coefficients
    if(validationFlag){
        validateSolverCoefficients();
    }

    freeCUDAMemory();
}

// Represent solver parameters in matrix format
void FDTDEngineCPU::solverParamsInMat(){

    int data_idx = 0;

    // Update solver coefficients
    for(int row_idx = 0; row_idx<frameH-2; row_idx++){
        for(int col_idx = 0; col_idx<frameW-2; col_idx++){
            
            data_idx = row_idx * (frameW-2) + col_idx;

            N_out_mat[row_idx][col_idx]                         = h_N_out[data_idx];
            N_in_mat[row_idx][col_idx]                          = h_N_in[data_idx];
            minVxBeta_mat[row_idx][col_idx]                     = h_minVxBeta[data_idx];
            minVyBeta_mat[row_idx][col_idx]                     = h_minVyBeta[data_idx];
            maxVxSigmaPrimedt_mat[row_idx][col_idx]             = h_maxVxSigmaPrimedt[data_idx];
            maxVySigmaPrimedt_mat[row_idx][col_idx]             = h_maxVySigmaPrimedt[data_idx];
            betaVxSqr_dt_inv_rho_inv_ds_mat[row_idx][col_idx]   = h_betaVxSqr_dt_inv_rho_inv_ds[data_idx];
            betaVySqr_dt_inv_rho_inv_ds_mat[row_idx][col_idx]   = h_betaVySqr_dt_inv_rho_inv_ds[data_idx];
            pressureSigmaPrimedt_mat[row_idx][col_idx]          = h_pressureSigmaPrimedt[data_idx];
            areWeNotExcitationVx_mat[row_idx][col_idx]          = h_areWeNotExcitationVx[data_idx];
            areWeNotExcitationVy_mat[row_idx][col_idx]          = h_areWeNotExcitationVy[data_idx];
            excitationWeightVx_mat[row_idx][col_idx]            = h_excitationWeightVx[data_idx];
            excitationWeightVy_mat[row_idx][col_idx]            = h_excitationWeightVy[data_idx];
            xor_val1_mat[row_idx][col_idx]                      = h_xor_val1[data_idx];
            xor_val2_mat[row_idx][col_idx]                      = h_xor_val2[data_idx];
            xor_val3_mat[row_idx][col_idx]                      = h_xor_val3[data_idx];
            xor_val4_mat[row_idx][col_idx]                      = h_xor_val4[data_idx];
        
        }
    }

    for(int row_idx = 0; row_idx<frameH; row_idx++){
        for(int col_idx = 0; col_idx<frameW; col_idx++){

            data_idx = row_idx * (frameW) + col_idx;

            // Update z_inv
            z_inv_mat[row_idx][col_idx] = h_z_inv[data_idx];

            // Update PV_N
            PV_N[row_idx][col_idx].Pr = h_PV_N[data_idx].Pr;
            PV_N[row_idx][col_idx].Vx = h_PV_N[data_idx].Vx;
            PV_N[row_idx][col_idx].Vy = h_PV_N[data_idx].Vy;
            PV_N[row_idx][col_idx].cell_type = h_PV_N[data_idx].cell_type;

            // Update PV_Nplus1
            PV_Nplus1[row_idx][col_idx].Pr        = h_PV_N[data_idx].Pr;
            PV_Nplus1[row_idx][col_idx].Vx        = h_PV_N[data_idx].Vx;
            PV_Nplus1[row_idx][col_idx].Vy        = h_PV_N[data_idx].Vy;
            PV_Nplus1[row_idx][col_idx].cell_type = h_PV_N[data_idx].cell_type;

            // Update depth components - depthP, depthX and depthY
            depthP[row_idx][col_idx] = h_depthP[data_idx];
            depthX[row_idx][col_idx] = h_depthX[data_idx];
            depthY[row_idx][col_idx] = h_depthY[data_idx];

        }
    }
}

// Copy the solver parameters from device to host
void FDTDEngineCPU::copySolverParamsFromDeviceToHost(){

    cudaMemcpy(h_N_out.data(),                       N_out,                       (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_N_in.data(),                        N_in,                        (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_minVxBeta.data(),                   minVxBeta,                   (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_minVyBeta.data(),                   minVyBeta,                   (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxVxSigmaPrimedt.data(),           maxVxSigmaPrimedt,           (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxVySigmaPrimedt.data(),           maxVySigmaPrimedt,           (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_betaVxSqr_dt_inv_rho_inv_ds.data(), betaVxSqr_dt_inv_rho_inv_ds, (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_betaVySqr_dt_inv_rho_inv_ds.data(), betaVySqr_dt_inv_rho_inv_ds, (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pressureSigmaPrimedt.data(),        pressureSigmaPrimedt,        (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_areWeNotExcitationVx.data(),        areWeNotExcitationVx,        (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_areWeNotExcitationVy.data(),        areWeNotExcitationVy,        (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_excitationWeightVx.data(),          excitationWeightVx,          (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_excitationWeightVy.data(),          excitationWeightVy,          (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xor_val1.data(),                    xor_val1,                    (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xor_val2.data(),                    xor_val2,                    (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xor_val3.data(),                    xor_val3,                    (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xor_val4.data(),                    xor_val4,                    (frameH-2)*(frameW-2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z_inv.data(),                       z_inv,                       (frameH)*(frameW)*sizeof(float), cudaMemcpyDeviceToHost);

    thrust::copy(_fdtdSolver.PV_N.begin(), _fdtdSolver.PV_N.end(), h_PV_N.begin());
    thrust::copy(_fdtdSolver.PV_Nplus1.begin(), _fdtdSolver.PV_Nplus1.end(), h_PV_Nplus1.begin());

    thrust::copy(_fdtdSolver.depthP.begin(), _fdtdSolver.depthP.end(), h_depthP.begin());
    thrust::copy(_fdtdSolver.depthX.begin(), _fdtdSolver.depthX.end(), h_depthX.begin());
    thrust::copy(_fdtdSolver.depthY.begin(), _fdtdSolver.depthY.end(), h_depthY.begin());
    
    
}

// Free the CUDA memory
void FDTDEngineCPU::freeCUDAMemory(){

    cudaFree(N_out);
    cudaFree(N_in);
    cudaFree(minVxBeta);
    cudaFree(minVyBeta);
    cudaFree(maxVxSigmaPrimedt);
    cudaFree(maxVySigmaPrimedt);
    cudaFree(betaVxSqr_dt_inv_rho_inv_ds);
    cudaFree(betaVySqr_dt_inv_rho_inv_ds);
    cudaFree(pressureSigmaPrimedt);
    cudaFree(areWeNotExcitationVx);
    cudaFree(areWeNotExcitationVy);
    cudaFree(excitationWeightVx);
    cudaFree(excitationWeightVy);
    cudaFree(xor_val1);
    cudaFree(xor_val2);
    cudaFree(xor_val3);
    cudaFree(xor_val4);
    cudaFree(z_inv);
}

/******** FDTD ENGINE ********/
void FDTDEngineCPU::startSolverEngine(){

    using Clock = chrono::high_resolution_clock;

    // Resize the wall displacement vectors
    w_prev.resize(frameH, vector<float>(frameW, 0.0f));
    w_next.resize(frameH, vector<float>(frameW, 0.0f));
    z_prev.resize(frameH, vector<float>(frameW, 0.0f));
    z_next.resize(frameH, vector<float>(frameW, 0.0f));
    record_pressure.resize(_fdtdSolver.num_steps, 0.0f);

    // Resize wave solver params
    CxVx.resize(frameH-2, vector<float>(frameW-2, 0.0f));
    CyVy.resize(frameH-2, vector<float>(frameW-2, 0.0f));
    CxP.resize(frameH-2, vector<float>(frameW-2, 0.0f));
    CyP.resize(frameH-2, vector<float>(frameW-2, 0.0f));

    Vb_alphaX.resize(frameH-2, vector<float>(frameW-2, 0.0f));
    Vb_alphaY.resize(frameH-2, vector<float>(frameW-2, 0.0f));

    // Determine excitation velocity using a vocal fold model or an excitation function 
    // Get the excitation velocity using an excitation function
    thrust::host_vector<float> excitation_velocity = _fdtdSolver.getExcitationVelocity();

    // For vocal fold model set the vocal fold parameters
    TwoMassModel twoMassModel;
    twoMassModel.setTwoMassModelParams();

    // Declare the excitation velocity that will be injected into the tube model at each time step
    float exeV = 0.0f;

    // Start the clock
    auto t0 = Clock::now();
    
    // Start the solver engine
    for (int t_step = 0; t_step < _fdtdSolver.num_steps; t_step++){

        // STEP1: Get the excitation velocity for the current time step
        if (_fdtdSolver.source_type == 2){

            // Retrieve pressure from the tube start postion [supra-glottal pressure]
            float p1 = PV_N[_fdtdSolver.excitationY][_fdtdSolver.excitationX].Pr;
            twoMassModel.setSupraGlottalPressure(p1);

            // Run the vocal fold model
            twoMassModel.runTwoMassModel(_fdtdSolver.srate_mul);
            exeV = twoMassModel.getVolumeVelocity();
        }    
        else
            exeV = excitation_velocity[t_step];

        // STEP2: Assign w_next and z_next to the w_prev and z_prev
        swap(w_prev, w_next);
        swap(z_prev, z_next);

        // STEP3: Update pressure and wall displacement
        update_wall_pres_and_wall_disp(t_step);

        // STEP4: Update velocity
        update_velocity(exeV);

        // STEP5: Handle the pressure and velocity for border cells
        update_border_cells();
        
        // STEP6: Copy next time-step data to the current time-step
        PV_N = PV_Nplus1;
    }

    // Stop the clock
    auto t1 = Clock::now();
    chrono::duration<double> simulation_duration = t1 - t0;

    // Print the simulation duration
    cout << "======================================" <<endl;
	cout << "Elapsed time [in seconds]: " << simulation_duration.count() << " s" << endl;
    cout << "======================================" <<endl;

    // Print to validate the acoustic parameters - P, Vx, Vy and cell_types
    // printGridCellComponents();
    // printDepthComponents();

    savePressureSamples();
}

void FDTDEngineCPU::update_wall_pres_and_wall_disp(int t_step){

    // [For 2D model]
    // Calculate del(V)/del(x) = (dVx/dx + dVy/dy) 
    // CxVx = dVx = V_curr - V_left,   where Vx = velocity along the x-direction
    // CyVy = dVy = V_curr - V_down, where Vy = velocity along the y-direction

    // [For 2.5D model]
    // Calculate del(DV) = d (depthX * Vx) + d (depthY * Vx)

    for(int row_idx=0; row_idx<frameH-2; row_idx++){
        for(int col_idx=0; col_idx<frameW-2; col_idx++){

            
            CxVx[row_idx][col_idx] = PV_N[row_idx+1][col_idx+1].Vx * depthX[row_idx+1][col_idx+1] - 
                                     PV_N[row_idx+1][col_idx].Vx   * depthX[row_idx+1][col_idx];

            CyVy[row_idx][col_idx] = PV_N[row_idx+1][col_idx+1].Vy * depthY[row_idx+1][col_idx+1] - 
                                     PV_N[row_idx+2][col_idx+1].Vy * depthY[row_idx+2][col_idx+1];
                            
            float numerator = ((PV_N[row_idx+1][col_idx+1].Pr * depthP[row_idx+1][col_idx+1]) - 
                              (_fdtdSolver.rho_sqrC_dt_inv_ds * (CxVx[row_idx][col_idx] + CyVy[row_idx][col_idx])) - 
                              (_fdtdSolver.rho_sqrC_dt * w_prev[row_idx+1][col_idx+1]));

            float denominator = ((1 + pressureSigmaPrimedt_mat[row_idx][col_idx]) * depthP[row_idx+1][col_idx+1]);

            // Update pressure at the next time step
            PV_Nplus1[row_idx+1][col_idx+1].Pr = numerator/denominator;
                                                 
            // Enforce Dirichlet boundary condition
            if(_fdtdSolver.mouth_radiation_condition && PV_Nplus1[row_idx+1][col_idx+1].cell_type == cell_noPressure)
                PV_Nplus1[row_idx+1][col_idx+1].Pr = 0;
        }
    }

    // Record pressure samples at the listener's position [listenerY, listenerX]
    record_pressure[t_step] = PV_Nplus1[listenerY][listenerX].Pr;

    // Compute w_next
    if(_fdtdSolver.fdtd_solver_type){
        
        for(int row_idx=0; row_idx<frameH; row_idx++){
            for(int col_idx=0; col_idx<frameW; col_idx++){

                z_next[row_idx][col_idx] = z_prev[row_idx][col_idx] + w_prev[row_idx][col_idx] * _fdtdSolver.dt;

                float num = (_fdtdSolver.Mw * w_prev[row_idx][col_idx]) + 
                            (_fdtdSolver.wall_pressure_coupling_coeff * _fdtdSolver.dt * PV_Nplus1[row_idx][col_idx].Pr) -
                            (_fdtdSolver.dt * _fdtdSolver.Kw * z_next[row_idx][col_idx]);

                float den = _fdtdSolver.Mw + (_fdtdSolver.dt * _fdtdSolver.Bw);

                w_next[row_idx][col_idx] = num/den;
            }
        }
    }
}

void FDTDEngineCPU::update_velocity(float exeV){

    for(int row_idx=0; row_idx<frameH-2; row_idx++){
        for(int col_idx=0; col_idx<frameW-2; col_idx++){ 

            // Calculate del(p)/del(x) or del(P)/del(y) 
            // CxP = P_right - P_curr
            // CyP = P_top - P_curr

            CxP[row_idx][col_idx] = PV_Nplus1[row_idx+1][col_idx+2].Pr - PV_Nplus1[row_idx+1][col_idx+1].Pr;
            CyP[row_idx][col_idx] = PV_Nplus1[row_idx][col_idx+1].Pr   - PV_Nplus1[row_idx+1][col_idx+1].Pr;
    
            PV_Nplus1[row_idx+1][col_idx+1].Vx = (minVxBeta_mat[row_idx][col_idx] * PV_N[row_idx+1][col_idx+1].Vx) - 
                                                 (betaVxSqr_dt_inv_rho_inv_ds_mat[row_idx][col_idx] * CxP[row_idx][col_idx]);

            PV_Nplus1[row_idx+1][col_idx+1].Vy = (minVyBeta_mat[row_idx][col_idx] * PV_N[row_idx+1][col_idx+1].Vy) - 
                                                 (betaVySqr_dt_inv_rho_inv_ds_mat[row_idx][col_idx] * CyP[row_idx][col_idx]);

            // Add source velocity
            PV_Nplus1[row_idx+1][col_idx+1].Vx = (PV_Nplus1[row_idx+1][col_idx+1].Vx) + 
                                                 (exeV * excitationWeightVx_mat[row_idx][col_idx] * maxVxSigmaPrimedt_mat[row_idx][col_idx]);

            PV_Nplus1[row_idx+1][col_idx+1].Vy = (PV_Nplus1[row_idx+1][col_idx+1].Vy) +
                                                 (exeV * excitationWeightVy_mat[row_idx][col_idx] * maxVySigmaPrimedt_mat[row_idx][col_idx]);


            // Update Vb_alphaX and Vb_alphaY
            Vb_alphaX[row_idx][col_idx] = (xor_val2_mat[row_idx][col_idx] * PV_Nplus1[row_idx+1][col_idx+1].Pr * N_out_mat[row_idx][col_idx]) -
                                          (xor_val1_mat[row_idx][col_idx] * PV_Nplus1[row_idx+1][col_idx+2].Pr * N_in_mat[row_idx][col_idx]);

            Vb_alphaY[row_idx][col_idx] = (xor_val4_mat[row_idx][col_idx] * PV_Nplus1[row_idx+1][col_idx+1].Pr * N_out_mat[row_idx][col_idx]) -
                                          (xor_val3_mat[row_idx][col_idx] * PV_Nplus1[row_idx][col_idx+1].Pr   * N_in_mat[row_idx][col_idx]);

            Vb_alphaX[row_idx][col_idx] = Vb_alphaX[row_idx][col_idx] * areWeNotExcitationVx_mat[row_idx][col_idx] * z_inv_mat[row_idx+1][col_idx+1];
            Vb_alphaY[row_idx][col_idx] = Vb_alphaY[row_idx][col_idx] * areWeNotExcitationVy_mat[row_idx][col_idx] * z_inv_mat[row_idx+1][col_idx+1];

            // Update PV_Nplus1.Vx and PV_Nplus1.Vy
            PV_Nplus1[row_idx+1][col_idx+1].Vx += (maxVxSigmaPrimedt_mat[row_idx][col_idx] * Vb_alphaX[row_idx][col_idx]);
            PV_Nplus1[row_idx+1][col_idx+1].Vy += (maxVySigmaPrimedt_mat[row_idx][col_idx] * Vb_alphaY[row_idx][col_idx]);

            PV_Nplus1[row_idx+1][col_idx+1].Vx /= (minVxBeta_mat[row_idx][col_idx] + maxVxSigmaPrimedt_mat[row_idx][col_idx]);
            PV_Nplus1[row_idx+1][col_idx+1].Vy /= (minVyBeta_mat[row_idx][col_idx] + maxVySigmaPrimedt_mat[row_idx][col_idx]);

        }
    } 
     
}

void FDTDEngineCPU::update_border_cells(){

    for(int row_idx=0; row_idx < frameH; row_idx++){
        PV_Nplus1[row_idx][0].Pr = 0;
        PV_Nplus1[row_idx][0].Vx = 0;
        PV_Nplus1[row_idx][0].Vy = 0;

        PV_Nplus1[row_idx][frameW-1].Pr = 0;
        PV_Nplus1[row_idx][frameW-1].Vx = 0;
        PV_Nplus1[row_idx][frameW-1].Vy = 0;
    }

    for(int col_idx=0; col_idx < frameW; col_idx++){
        PV_Nplus1[0][col_idx].Pr = 0;
        PV_Nplus1[0][col_idx].Vx = 0;
        PV_Nplus1[0][col_idx].Vy = 0;

        PV_Nplus1[frameH-1][col_idx].Pr = 0;
        PV_Nplus1[frameH-1][col_idx].Vx = 0;
        PV_Nplus1[frameH-1][col_idx].Vy = 0;
    }
}

void FDTDEngineCPU::savePressureSamples(){

    //  Open an ofstream
    ofstream out("../../results/recorded-pressure.txt");
    if (!out) {
        std::cerr << "Error: could not open recorded-pressure.txt for writing\n";
        std::exit(1);
    }

    // Write one value per line
    for (size_t i = 0; i < record_pressure.size(); ++i) {
        out << record_pressure[i] << "\n";
    }

    out.close();
}

/******** VALIDATE FDTD SOLVER ********/

// Validate the solver coefficients by printing in matrix format
void FDTDEngineCPU::validateSolverCoefficients(){
    
    int num_rows = frameH-2;
    int num_cols = frameW-2;

    printMatrixInCSV(N_out_mat, "N_out", num_rows, num_cols);
    printMatrixInCSV(N_in_mat, "N_in", num_rows, num_cols);
    printMatrixInCSV(minVxBeta_mat, "minVxBeta", num_rows, num_cols);
    printMatrixInCSV(minVyBeta_mat, "minVyBeta", num_rows, num_cols);
    printMatrixInCSV(maxVxSigmaPrimedt_mat, "maxVxSigmaPrimedt", num_rows, num_cols);
    printMatrixInCSV(maxVySigmaPrimedt_mat, "maxVySigmaPrimedt", num_rows, num_cols);
    printMatrixInCSV(betaVxSqr_dt_inv_rho_inv_ds_mat, "betaVxSqr_dt_inv_rho_inv_ds", num_rows, num_cols);
    printMatrixInCSV(betaVySqr_dt_inv_rho_inv_ds_mat, "betaVySqr_dt_inv_rho_inv_ds", num_rows, num_cols);
    printMatrixInCSV(pressureSigmaPrimedt_mat, "pressureSigmaPrimedt", num_rows, num_cols);
    printMatrixInCSV(areWeNotExcitationVx_mat, "areWeNotExcitationVx", num_rows, num_cols);
    printMatrixInCSV(areWeNotExcitationVy_mat, "areWeNotExcitationVy", num_rows, num_cols);
    printMatrixInCSV(excitationWeightVx_mat, "excitationWeightVx", num_rows, num_cols);
    printMatrixInCSV(excitationWeightVy_mat, "excitationWeightVy", num_rows, num_cols);
    printMatrixInCSV(xor_val1_mat, "xor_val1", num_rows, num_cols);
    printMatrixInCSV(xor_val2_mat, "xor_val2", num_rows, num_cols);
    printMatrixInCSV(xor_val3_mat, "xor_val3", num_rows, num_cols);
    printMatrixInCSV(xor_val4_mat, "xor_val4", num_rows, num_cols);
    printMatrixInCSV(z_inv_mat, "z_inv", num_rows, num_cols);
}

// Print data in matrix format
void FDTDEngineCPU::printMatrixInCSV(vector<vector<float>> dataToPrint, string file_name, int num_rows, int num_cols){

    ofstream print_data;
    string file_path = "../../data/" + file_name + ".csv";

    print_data.open(file_path);

    for(int row_idx=0; row_idx < num_rows; row_idx++){
        for(int col_idx=0; col_idx < num_cols; col_idx++){

            print_data << dataToPrint[row_idx][col_idx] << ',';
        }

        print_data << endl;
    }
    print_data.close();
}

// Print Pressure, velocty components and cell_types
void FDTDEngineCPU::printGridCellComponents(){

    ofstream print_pressure;
    ofstream print_vx;
    ofstream print_vy;
    ofstream print_cell_types;

    print_pressure.open("../../data/Pressure.csv");
    print_vx.open("../../data/Vx.csv");
    print_vy.open("../../data/vy.csv");
    print_cell_types.open("../../data/cell_types.csv");

    for (int row_idx=0; row_idx<frameH; row_idx++){
        for (int col_idx=0; col_idx<frameW; col_idx++){
            print_pressure << PV_Nplus1[row_idx][col_idx].Pr << ',';
            print_vx << PV_Nplus1[row_idx][col_idx].Vx << ',';
            print_vy << PV_Nplus1[row_idx][col_idx].Vy << ',';
            print_cell_types << PV_Nplus1[row_idx][col_idx].cell_type << ',';
        }

        print_pressure << endl;
        print_vx << endl;
        print_vy << endl;
        print_cell_types << endl;
    }

    print_pressure.close();
    print_vx.close();
    print_vy.close();
    print_cell_types.close();

}

// Print depth components - depthP, depthX and depthY
void FDTDEngineCPU::printDepthComponents(){

    ofstream print_depthP;
    ofstream print_depthX;
    ofstream print_depthY;

    print_depthP.open("../../data/depthP.csv");
    print_depthX.open("../../data/depthX.csv");
    print_depthY.open("../../data/depthY.csv");

    for (int row_idx=0; row_idx<frameH; row_idx++){
        for (int col_idx=0; col_idx<frameW; col_idx++){
            print_depthP << depthP[row_idx][col_idx]<< ',';
            print_depthX << depthX[row_idx][col_idx] << ',';
            print_depthY << depthY[row_idx][col_idx] << ',';
        }

        print_depthP << endl;
        print_depthX << endl;
        print_depthY << endl;
    }

    print_depthP.close();
    print_depthX.close();
    print_depthY.close();
}