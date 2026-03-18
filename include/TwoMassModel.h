/*
 * TwoMassModelParams.h
 * 
 * Define two Mass vocal fold model parameters
 * 
 */

#ifndef TWOMASSMODEL_H_
#define TWOMASSMODEL_H_

#include <cmath>
#include "SimulationUnits.h"

class TwoMassModel{

    private:

        float mu;       // Air viscosity
        float rho;      // Air density

        float q_factor; // Pitch factor
        float gs;       // Dimensionless damping factor

        // Vocal fold mass
        float m1; 
        float m2;

        // Thickness of each mass
        float d1;
        float d2;

        // Linear spring stiffness
        float k1;
        float k2;
        float kc;

        // Non-linear spring stiffness
        float etak1;
        float etak2;

        // Linear stiffness during vocal fold collision
        float h1;
        float h2;

        // Non-linear stiffness during vocal fold collision
        float etah1;
        float etah2;

        // Cross sectional area of glottal slit at rest
        float Ag0;
        float Ag01;
        float Ag02;

        // Vocal cord effective length
        float lg;

        // Damping ratio
        float zeta1_open;
        float zeta2_open;
        float zeta1_close;
        float zeta2_close;

        // Viscous resitance
        float r1_open;
        float r2_open;
        float r1_close;
        float r2_close;

        // Input area to the vocal tract
        float A1;

        // Change in glottal area
        float Ag1;
        float Ag2;

        // Sub-glottal(Lungs) pressure
        float ps;

        float p1;

        // Force on mass m1 and m2
        float f_mass1;
        float f_mass2;

        // Pressure on mass m1 and m2
        float p_mass1;
        float p_mass2;

        // Glottal resitive parameters
        float Rv1;
        float Rv2;

        // Glottal inductive parameters
        float Lg1;
        float Lg2;

        // Volume velocity
        float ug_old;
        float ug_curr;
        float ug_next;

        // Latteral displacement of vocal fold masses
        float x1_old;
        float x1_curr;
        float x1_next;

        float x2_old;
        float x2_curr;
        float x2_next;

        // Matrix Parameters 
        float a11;
        float a12;
        float a21;
        float a22;
        float b1;
        float b2;

        float s1_prime;
        float s2_prime;

    public:
        void setTwoMassModelParams();
        void runTwoMassModel(int srate_mul);
        void setSupraGlottalPressure(float pressure) { p1 = pressure; }
        float getVolumeVelocity() { return ug_next; }
};

#endif // TWOMASSMODELPARAMS_H_