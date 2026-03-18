/*
 * DeviceSymbols.h
 * 
 * Declare FDTD simulation device constants and variables
 * 
 */

 #ifndef DEVICESYMBOLS_H_
 #define DEVICESYMBOLS_H_

#ifdef NUM_CELL_TYPES
#undef NUM_CELL_TYPES
#endif

#define NUM_CELL_TYPES  13

#ifndef BUILD_FDTD_CONSTANTS_CU

    // Declare device constants for FDTD simulation
    extern __constant__ float d_sound_speed;
    extern __constant__ float d_rho;
    extern __constant__ float d_rho_sqrC_dt_inv_ds;
    extern __constant__ float d_rho_sqrC_dt;
    extern __constant__ float d_wall_pressure_coupling_coeff;
    extern __constant__ float d_Mw;
    extern __constant__ float d_Bw;
    extern __constant__ float d_Kw;
    extern __constant__ float d_ds;
    extern __constant__ float d_dt;

    extern __constant__ float d_beta[NUM_CELL_TYPES];
    extern __constant__ float d_sigma_prime_dt[NUM_CELL_TYPES];
    extern __constant__ float d_source_direction[4];

    extern __constant__ float air_normal_component[4];
    extern __constant__ float wall_normal_component[4];

#else
    #define BUILD_FDTD_CONSTANTS_CU
#endif

#endif // DEVICESYMBOLS_H_