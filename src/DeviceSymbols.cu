#define BUILD_FDTD_CONSTANTS_CU

#include "../include/DeviceSymbols.h"

__constant__ float d_sound_speed;
__constant__ float d_rho;
__constant__ float d_rho_sqrC_dt_inv_ds;
__constant__ float d_rho_sqrC_dt;
__constant__ float d_wall_pressure_coupling_coeff;
__constant__ float d_Mw;
__constant__ float d_Bw;
__constant__ float d_Kw;
__constant__ float d_ds;
__constant__ float d_dt;

__constant__ float d_beta[NUM_CELL_TYPES];
__constant__ float d_sigma_prime_dt[NUM_CELL_TYPES];
__constant__ float d_source_direction[4];

__constant__ float air_normal_component[]  = {0, 1.0, 1.0, 0.707106};
__constant__ float wall_normal_component[] = {1.0, 1.0, 1.0, 0}; 
