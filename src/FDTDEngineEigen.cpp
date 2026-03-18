
#include "../include/FDTDEngineEigen.h"

FDTDEngineEigen::FDTDEngineEigen(FDTDEngineCPU& fdtdEngineCPU, FDTDSolver& fdtdSolver, bool validationFlag)
                                :_fdtdEngineCPU(fdtdEngineCPU), _fdtdSolver(fdtdSolver)
{

    // Get the computational domain size
    frameH = _fdtdSolver.frameH;
    frameW = _fdtdSolver.frameW;

    // Get the listener position
    listenerX = _fdtdSolver.listenerX;
    listenerY = _fdtdSolver.listenerY;

    // Initialize the solver coefficients
    initializeSolverCoeffs();

    // Assign solver coefficients' value in FDTDEngineCPU to FDTDEngineEigen
    assignSolverCoeffs();

    // Validate solver coefficients
    if (validationFlag){
        validateSolverCoefficients();
    }
}

void FDTDEngineEigen::initializeSolverCoeffs(){

    int num_rows = frameH - 2;
    int num_cols = frameW - 2;

    CxVx.setZero(num_rows, num_cols);
    CyVy.setZero(num_rows, num_cols);
    CxP.setZero(num_rows, num_cols);
    CyP.setZero(num_rows, num_cols);
    Vb_alphaX.setZero(num_rows, num_cols);
    Vb_alphaY.setZero(num_rows, num_cols);

    N_out_mat.setZero(num_rows, num_cols);
    N_in_mat.setZero(num_rows, num_cols);
    minVxBeta_mat.setZero(num_rows, num_cols);
    minVyBeta_mat.setZero(num_rows, num_cols);
    maxVxSigmaPrimedt_mat.setZero(num_rows, num_cols);
    maxVySigmaPrimedt_mat.setZero(num_rows, num_cols);
    betaVxSqr_dt_inv_rho_inv_ds_mat.setZero(num_rows, num_cols);
    betaVySqr_dt_inv_rho_inv_ds_mat.setZero(num_rows, num_cols);
    pressureSigmaPrimedt_mat.setZero(num_rows, num_cols);
    areWeNotExcitationVx_mat.setZero(num_rows, num_cols);
    areWeNotExcitationVy_mat.setZero(num_rows, num_cols);
    excitationWeightVx_mat.setZero(num_rows, num_cols);
    excitationWeightVy_mat.setZero(num_rows, num_cols);
    xor_val1_mat.setZero(num_rows, num_cols);
    xor_val2_mat.setZero(num_rows, num_cols);
    xor_val3_mat.setZero(num_rows, num_cols);
    xor_val4_mat.setZero(num_rows, num_cols);
    z_inv_mat.setZero(frameH, frameW);

    w_next.setZero(frameH, frameW);
    w_prev.setZero(frameH, frameW);
    z_next.setZero(frameH, frameW);
    z_prev.setZero(frameH, frameW);

    Pr_N.setZero(frameH, frameW);
    Pr_Nplus1.setZero(frameH, frameW);
    Vx_N.setZero(frameH, frameW);
    Vx_Nplus1.setZero(frameH, frameW);
    Vy_N.setZero(frameH, frameW);
    Vy_Nplus1.setZero(frameH, frameW);

    cell_type.setOnes(frameH, frameW); // <- Declare all the cells as air cells

    depthP.setZero(frameH, frameW);
    depthX.setZero(frameH, frameW);
    depthY.setZero(frameH, frameW);

    record_pressure.setZero(_fdtdSolver.num_steps, 1);
}

void FDTDEngineEigen::assignSolverCoeffs(){

    for(int row_idx=0; row_idx < frameH-2; row_idx++){
        for(int col_idx=0; col_idx < frameW-2; col_idx++){

            N_out_mat(row_idx, col_idx)                        = _fdtdEngineCPU.N_out_mat[row_idx][col_idx];
            N_in_mat(row_idx, col_idx)                         = _fdtdEngineCPU.N_in_mat[row_idx][col_idx];
            minVxBeta_mat(row_idx, col_idx)                    = _fdtdEngineCPU.minVxBeta_mat[row_idx][col_idx];
            minVyBeta_mat(row_idx, col_idx)                    = _fdtdEngineCPU.minVyBeta_mat[row_idx][col_idx];
            maxVxSigmaPrimedt_mat(row_idx, col_idx)            = _fdtdEngineCPU.maxVxSigmaPrimedt_mat[row_idx][col_idx];
            maxVySigmaPrimedt_mat(row_idx, col_idx)            = _fdtdEngineCPU.maxVySigmaPrimedt_mat[row_idx][col_idx];
            betaVxSqr_dt_inv_rho_inv_ds_mat(row_idx, col_idx)  = _fdtdEngineCPU.betaVxSqr_dt_inv_rho_inv_ds_mat[row_idx][col_idx];
            betaVySqr_dt_inv_rho_inv_ds_mat(row_idx, col_idx)  = _fdtdEngineCPU.betaVySqr_dt_inv_rho_inv_ds_mat[row_idx][col_idx];
            pressureSigmaPrimedt_mat(row_idx, col_idx)         = _fdtdEngineCPU.pressureSigmaPrimedt_mat[row_idx][col_idx];
            areWeNotExcitationVx_mat(row_idx, col_idx)         = _fdtdEngineCPU.areWeNotExcitationVx_mat[row_idx][col_idx];
            areWeNotExcitationVy_mat(row_idx, col_idx)         = _fdtdEngineCPU.areWeNotExcitationVy_mat[row_idx][col_idx];
            excitationWeightVx_mat(row_idx, col_idx)           = _fdtdEngineCPU.excitationWeightVx_mat[row_idx][col_idx];
            excitationWeightVy_mat(row_idx, col_idx)           = _fdtdEngineCPU.excitationWeightVy_mat[row_idx][col_idx];
            xor_val1_mat(row_idx, col_idx)                     = _fdtdEngineCPU.xor_val1_mat[row_idx][col_idx];
            xor_val2_mat(row_idx, col_idx)                     = _fdtdEngineCPU.xor_val2_mat[row_idx][col_idx];
            xor_val3_mat(row_idx, col_idx)                     = _fdtdEngineCPU.xor_val3_mat[row_idx][col_idx];
            xor_val4_mat(row_idx, col_idx)                     = _fdtdEngineCPU.xor_val4_mat[row_idx][col_idx];

        }
    }

    for(int row_idx=0; row_idx < frameH; row_idx++){
        for(int col_idx=0; col_idx < frameW; col_idx++){

            depthP(row_idx, col_idx) = _fdtdEngineCPU.depthP[row_idx][col_idx];
            depthX(row_idx, col_idx) = _fdtdEngineCPU.depthX[row_idx][col_idx];
            depthY(row_idx, col_idx) = _fdtdEngineCPU.depthY[row_idx][col_idx];
            z_inv_mat(row_idx, col_idx) = _fdtdEngineCPU.z_inv_mat[row_idx][col_idx];

            cell_type(row_idx, col_idx) = _fdtdEngineCPU.PV_N[row_idx][col_idx].cell_type;

        }
    }
}

/******** FDTD ENGINE ********/
void FDTDEngineEigen::startSolverEngine(){

    using Clock = chrono::high_resolution_clock;

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
            float p1 = Pr_N(_fdtdSolver.excitationY, _fdtdSolver.excitationX);
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
        Pr_N = Pr_Nplus1;
        Vx_N = Vx_Nplus1;
        Vy_N = Vy_Nplus1;
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

void FDTDEngineEigen::update_border_cells(){

    // left border (column 0)
    Pr_Nplus1 .col(0).setZero();
    Vx_Nplus1 .col(0).setZero();
    Vy_Nplus1 .col(0).setZero();

    // right border (column frameW-1)
    Pr_Nplus1 .col(frameW-1).setZero();
    Vx_Nplus1 .col(frameW-1).setZero();
    Vy_Nplus1 .col(frameW-1).setZero();

    // Top border (row 0)
    Pr_Nplus1 .row(0).setZero();
    Vx_Nplus1 .row(0).setZero();
    Vy_Nplus1 .row(0).setZero();

    // Bottom border (row frameH-1)
    Pr_Nplus1 .row(frameH-1).setZero();
    Vx_Nplus1 .row(frameH-1).setZero();
    Vy_Nplus1 .row(frameH-1).setZero();

}

void FDTDEngineEigen::update_wall_pres_and_wall_disp(int t_step){

    // [For 2D model]
    // Calculate del(V)/del(x) = (dVx/dx + dVy/dy) 
    // CxVx = dVx = V_curr - V_left,   where Vx = velocity along the x-direction
    // CyVy = dVy = V_curr - V_down, where Vy = velocity along the y-direction

    // [For 2.5D model]
    // Calculate del(DV) = d (depthX * Vx) + d (depthY * Vx)

    // Usage of M.block(startRow, startCol, numRows, numCols)

    int H = frameH-2;
    int W = frameW-2;

    auto Pr_curr = Pr_N.block(1, 1, H, W);
    auto Vx_curr = Vx_N.block(1, 1, H, W);
    auto Vx_left = Vx_N.block(1, 0, H, W);
    auto Vy_curr = Vy_N.block(1, 1, H, W);
    auto Vy_down = Vy_N.block(2, 1, H, W);

    auto dp_curr = depthP.block(1, 1, H, W);
    auto dx_curr = depthX.block(1, 1, H, W);
    auto dx_left = depthX.block(1, 0, H, W);
    auto dy_curr = depthY.block(1, 1, H, W);
    auto dy_down = depthY.block(2, 1, H, W);

    auto w_prev_curr = w_prev.block(1, 1, H, W);

    CxVx.noalias() = Vx_curr.cwiseProduct(dx_curr) - Vx_left.cwiseProduct(dx_left);
    CyVy.noalias() = Vy_curr.cwiseProduct(dy_curr) - Vy_down.cwiseProduct(dy_down);

    auto numerator = Pr_curr.cwiseProduct(dp_curr) 
                     - (_fdtdSolver.rho_sqrC_dt_inv_ds * (CxVx + CyVy))
                     - (_fdtdSolver.rho_sqrC_dt * w_prev_curr);

    auto denominator = pressureSigmaPrimedt_mat.cwiseProduct(dp_curr) + dp_curr;

    Pr_Nplus1.block(1, 1, H, W).noalias() = numerator.cwiseQuotient(denominator);

    // Enforce Dirichlet boundary condition
    if(_fdtdSolver.mouth_radiation_condition){
        Eigen::MatrixXf mask_Mouth_end_cells = (cell_type.array() != cell_noPressure).cast<float>();
        Pr_Nplus1 = Pr_Nplus1.cwiseProduct(mask_Mouth_end_cells);
    }

    // Record pressure samples at the listener's position [listenerY, listenerX]
    record_pressure(t_step) = Pr_Nplus1(listenerY, listenerX);

    // Compute w_next
    if(_fdtdSolver.fdtd_solver_type){

        z_next.noalias() = z_prev + (_fdtdSolver.dt * w_prev);

        auto num = (_fdtdSolver.Mw * w_prev) + 
                    (_fdtdSolver.wall_pressure_coupling_coeff * _fdtdSolver.dt * Pr_Nplus1) -
                    (_fdtdSolver.dt * _fdtdSolver.Kw * z_next);

        auto den = _fdtdSolver.Mw + (_fdtdSolver.dt * _fdtdSolver.Bw );

        w_next.noalias() = num/den;
    }
}

void FDTDEngineEigen::update_velocity(float exeV){

    // Calculate del(p)/del(x) or del(P)/del(y) 
    // CxP = P_right - P_curr
    // CyP = P_top - P_curr

    int H = frameH-2;
    int W = frameW-2;

    auto Pr_curr  = Pr_Nplus1.block(1, 1, H, W);
    auto Pr_right = Pr_Nplus1.block(1, 2, H, W);
    auto Pr_top   = Pr_Nplus1.block(0, 1, H, W);

    auto Vx_prev = Vx_N.block(1, 1, H, W);
    auto Vy_prev = Vy_N.block(1, 1, H, W);

    CxP.noalias() = Pr_right - Pr_curr;
    CyP.noalias() = Pr_top - Pr_curr;
    
    Vx_Nplus1.block(1, 1, H, W).noalias() = minVxBeta_mat.cwiseProduct(Vx_prev) - 
                                            betaVxSqr_dt_inv_rho_inv_ds_mat.cwiseProduct(CxP);

    Vy_Nplus1.block(1, 1, H, W).noalias() = minVyBeta_mat.cwiseProduct(Vy_prev) - 
                                            betaVySqr_dt_inv_rho_inv_ds_mat.cwiseProduct(CyP);

    // Add source velocity
    Vx_Nplus1.block(1, 1, H, W) += (exeV * excitationWeightVx_mat).cwiseProduct(maxVxSigmaPrimedt_mat);
    Vy_Nplus1.block(1, 1, H, W) += (exeV * excitationWeightVy_mat).cwiseProduct(maxVySigmaPrimedt_mat);

    // Update Vb_alphaX and Vb_alphaY
    Vb_alphaX.noalias() = (xor_val2_mat.cwiseProduct(Pr_curr)).cwiseProduct(N_out_mat) - 
                          (xor_val1_mat.cwiseProduct(Pr_right)).cwiseProduct(N_in_mat);
    
    Vb_alphaY.noalias() = (xor_val4_mat.cwiseProduct(Pr_curr)).cwiseProduct(N_out_mat) - 
                          (xor_val3_mat.cwiseProduct(Pr_top)).cwiseProduct(N_in_mat);

    Vb_alphaX = (Vb_alphaX.cwiseProduct(areWeNotExcitationVx_mat)).cwiseProduct(z_inv_mat.block(1, 1, H, W));
    Vb_alphaY = (Vb_alphaY.cwiseProduct(areWeNotExcitationVy_mat)).cwiseProduct(z_inv_mat.block(1, 1, H, W));

    // Update Vx_Nplus1 and Vy_Nplus1
    Vx_Nplus1.block(1, 1, H, W) += maxVxSigmaPrimedt_mat.cwiseProduct(Vb_alphaX);
    Vy_Nplus1.block(1, 1, H, W) += maxVySigmaPrimedt_mat.cwiseProduct(Vb_alphaY);

    Vx_Nplus1.block(1, 1, H, W) = Vx_Nplus1.block(1, 1, H, W).cwiseQuotient(minVxBeta_mat + maxVxSigmaPrimedt_mat);
    Vy_Nplus1.block(1, 1, H, W) = Vy_Nplus1.block(1, 1, H, W).cwiseQuotient(minVyBeta_mat + maxVySigmaPrimedt_mat);
}

/******** VALIDATE FDTD SOLVER ********/

// Validate the solver coefficients by printing in matrix format
void FDTDEngineEigen::validateSolverCoefficients(){

    printMatrixInCSV("N_out",                       N_out_mat);
    printMatrixInCSV("N_in",                        N_in_mat);
    printMatrixInCSV("minVxBeta",                   minVxBeta_mat);
    printMatrixInCSV("minVyBeta",                   minVyBeta_mat);
    printMatrixInCSV("maxVxSigmaPrimedt",           maxVxSigmaPrimedt_mat);
    printMatrixInCSV("maxVySigmaPrimedt",           maxVySigmaPrimedt_mat);
    printMatrixInCSV("betaVxSqr_dt_inv_rho_inv_ds", betaVxSqr_dt_inv_rho_inv_ds_mat);
    printMatrixInCSV("betaVySqr_dt_inv_rho_inv_ds", betaVySqr_dt_inv_rho_inv_ds_mat);
    printMatrixInCSV("pressureSigmaPrimedt",        pressureSigmaPrimedt_mat);
    printMatrixInCSV("areWeNotExcitationVx",        areWeNotExcitationVx_mat);
    printMatrixInCSV("areWeNotExcitationVy",        areWeNotExcitationVy_mat);
    printMatrixInCSV("excitationWeightVx",          excitationWeightVx_mat);
    printMatrixInCSV("excitationWeightVy",          excitationWeightVy_mat);
    printMatrixInCSV("xor_val1",                    xor_val1_mat);
    printMatrixInCSV("xor_val2",                    xor_val2_mat);
    printMatrixInCSV("xor_val3",                    xor_val3_mat);
    printMatrixInCSV("xor_val4",                    xor_val4_mat);
    printMatrixInCSV("z_inv",                       z_inv_mat);

}

// Print data in matrix format
void FDTDEngineEigen::printMatrixInCSV(const string& file_name, const MatrixXf& data_mat){

    std::string file_path = "../../data/" + file_name + ".csv";
    std::ofstream out(file_path);
    
    if (!out) throw std::runtime_error("Cannot open " + file_path);

    Eigen::IOFormat csvFmt(Eigen::FullPrecision,    // print all digits
                           Eigen::DontAlignCols,    // no extra spacing
                           ",",                     // coeff separator
                           "\n");                   // row separator

    out << data_mat.format(csvFmt);
}

void FDTDEngineEigen::printMatrixInCSV(const string& file_name, const MatrixXi& data_mat){

    string file_path = "../../data/" + file_name + ".csv";
    ofstream out(file_path);
    
    if (!out) throw std::runtime_error("Cannot open " + file_name);

    Eigen::IOFormat csvFmt(
      Eigen::FullPrecision,    // print all digits
      Eigen::DontAlignCols,    // no extra spacing
      ",",                     // coeff separator
      "\n");                   // row separator

    out << data_mat.format(csvFmt);
}

void FDTDEngineEigen::savePressureSamples(){

    std::ofstream out("../../results/recorded-pressure.txt", std::ios::out | std::ios::trunc);

    if (!out) throw std::runtime_error("Could not open pressure.csv");

    for (int idx = 0; idx < record_pressure.size(); ++idx) {
        out << record_pressure(idx) << '\n';
    }
}

void FDTDEngineEigen::printGridCellComponents(){

    Eigen::IOFormat csvFmt(Eigen::FullPrecision,    // print all digits
                           Eigen::DontAlignCols,    // no extra spacing
                           ",",                     // coeff separator
                           "\n");                   // row separator
    
    // Print pressure - Pr
    ofstream pressureOut("../../data/pressure.csv", std::ios::out | std::ios::trunc);
    if (!pressureOut) throw std::runtime_error("Could not open pressure.csv");
    pressureOut << Pr_N.format(csvFmt);
    
    // Print Vx
    ofstream VxOut("../../data/Vx.csv", std::ios::out | std::ios::trunc);
    if (!VxOut) throw std::runtime_error("Could not open Vx.csv");
    VxOut << Vx_N.format(csvFmt);

    // Print Vy
    ofstream VyOut("../../data/Vy.csv", std::ios::out | std::ios::trunc);
    if (!VyOut) throw std::runtime_error("Could not open Vy.csv");
    VyOut << Vy_N.format(csvFmt);
    
    // Print cell_type
    ofstream celltypeOut("../../data/cell_type.csv", std::ios::out | std::ios::trunc);
    if (!celltypeOut) throw std::runtime_error("Could not open cell_type.csv");
    celltypeOut << cell_type.format(csvFmt);
}

void FDTDEngineEigen::printDepthComponents(){

    Eigen::IOFormat csvFmt(Eigen::FullPrecision,    // print all digits
                           Eigen::DontAlignCols,    // no extra spacing
                           ",",                     // coeff separator
                           "\n");                   // row separator

    // Print depthP
    ofstream depthpOut("../../data/depthP.csv", std::ios::out | std::ios::trunc);
    if (!depthpOut) throw std::runtime_error("Could not open depthP.csv");
    depthpOut << depthP.format(csvFmt);

    // Print depthX
    ofstream depthxOut("../../data/depthX.csv", std::ios::out | std::ios::trunc);
    if (!depthxOut) throw std::runtime_error("Could not open depthX.csv");
    depthxOut << depthX.format(csvFmt);

    // Print depthY
    ofstream depthyOut("../../data/depthY.csv", std::ios::out | std::ios::trunc);
    if (!depthyOut) throw std::runtime_error("Could not open depthY.csv");
    depthyOut << depthY.format(csvFmt);
    
}
