#include "../include/TwoMassModel.h"
#include <iostream>
// Set vocal fold parameters for the two-mass model
void TwoMassModel::setTwoMassModelParams(){

    mu = 1.86e-4 * (DYN * SECOND/(CENTIMETER*CENTIMETER));
    rho = 1.14 * (KILOGRAM / (METER * METER * METER));

    q_factor = 1.0f;
    gs  = 1.0f;

    m1 = (0.125f * GRAM)/q_factor;
    m2 = (0.025f * GRAM)/q_factor;

    d1 = (0.25f * CENTIMETER)/q_factor;
    d2 = (0.05f * CENTIMETER)/q_factor;

    k1 = 80000.0f * (DYN/CENTIMETER) * q_factor;
    k2 = 8000.0f  * (DYN/CENTIMETER) * q_factor;
    kc = 25000.0f * (DYN/CENTIMETER) * (q_factor*q_factor);

    etak1 = 100/(CENTIMETER * CENTIMETER);
    etak2 = 100/(CENTIMETER * CENTIMETER);

    h1 = 3 * k1;
    h2 = 3 * k2;

    etah1 = 500.0f / (CENTIMETER * CENTIMETER);
    etah2 = 500.0f / (CENTIMETER * CENTIMETER);

    Ag0 = 0.05f * (CENTIMETER * CENTIMETER);
    Ag01 = Ag0;
    Ag02 = Ag0;

    lg = 1.40f * CENTIMETER;
    
    zeta1_open = 0.2f;
    zeta2_open = 0.6f;
    zeta1_close = 1.1f;
    zeta2_close = 1.9f;

    r1_open  = (2 * zeta1_open  * sqrtf(k1 * m1)) / (gs * gs);
    r2_open  = (2 * zeta2_open  * sqrtf(k2 * m2)) / (gs * gs);
    r1_close = (2 * zeta1_close * sqrtf(k1 * m1)) / (gs * gs);
    r2_close = (2 * zeta2_close * sqrtf(k2 * m2)) / (gs * gs);

    A1 = 3.0f * (CENTIMETER * CENTIMETER);
    Ag1 = 0.0f;
    Ag2 = 0.0f;

    // In Pascal(7cm H2O) ==== 1cmH2O => 98.0665 => 784.532
    ps = 686.46f;

    p1 = 0.0f;

    f_mass1 = 0.0f;
    f_mass2 = 0.0f;
    p_mass1 = 0.0f;
    p_mass2 = 0.0f;

    Rv1 = 0.0f;
    Rv2 = 0.0f;
    Lg1 = 0.0f;
    Lg2 = 0.0f;

    ug_old = 0.0f;
    ug_curr = 0.0f;
    ug_next = 0.0f;

    x1_old = 0.0f;
    x1_curr = 0.0f;
    x1_next = 0.0f;

    x2_old = 0.0f;
    x2_curr = 0.0f;
    x2_next = 0.0f;

    a11 = 0.0f;
    a12 = 0.0f;
    a21 = 0.0f;
    a22 = 0.0f;
    b1 = 0.0f;
    b2 = 0.0f;

    s1_prime = 0.0f;
    s2_prime = 0.0f;
}

// Run the two-mass model simulation
void TwoMassModel::runTwoMassModel(int srate_mul){

    // Calculate srate
    float srate = 44100 * static_cast<float>(srate_mul); 

    // Calculate glottal areas
    Ag1 = Ag01 + 2*lg*x1_curr;
    Ag2 = Ag02 + 2*lg*x2_curr;

    // Use Equation 18 from IF72 to compute the force
    // Use Equation 17 from IF72 to compute the following parameters: Lg1, Lg2, Rv1, Rv2
    // Implement the conditional table of Equation 14 from IF72 (p-1244)

    // Calculate the parameters (Required to compute f_mass1 & f_mass2 ): Rv1, Lg1, Rv2, Lg2
    Rv1 = (12*mu*lg*lg*d1) / (Ag1*Ag1*Ag1);
    Lg1 = (rho*d1) / (Ag1);
    
    Rv2 = (12*mu*lg*lg*d2) / (Ag2*Ag2*Ag2);
    Lg2 = (rho*d2) / (Ag2);

    // Compute force f1 for mass m1
    if (x1_curr > -Ag01 / (2 * lg) && x2_curr > -Ag02 / (2 * lg)){

        p_mass1 = ps - 1.37*(rho / 2)*(ug_curr / Ag1)*(ug_curr / Ag1) - 0.5*(Rv1*ug_curr + Lg1 * srate * (ug_curr - ug_old));
        f_mass1 = p_mass1 * d1 * lg;
    }     
    else{
        p_mass1 = ps;
        f_mass1 = p_mass1 * d1 * lg;
    }

    // Compute force f2 for mass m2
    if (x1_curr > -Ag01 / (2 * lg)){  
        if (x2_curr > -Ag02 / (2 * lg)){
            p_mass2 = p_mass1 - (0.5*(Rv1 + Rv2)*ug_curr + (Lg1 + Lg2)*(ug_curr - ug_old)*srate) - (rho / 2 * ug_curr*ug_curr)*(1 / (Ag2*Ag2) - 1 / (Ag1*Ag1));      
            f_mass2 = p_mass2 * d2 * lg;
        }    
        else{
            p_mass2 = ps;
            f_mass2 = p_mass2 * d2 * lg;
        }
    }    
    else{
        p_mass2 = 0;
        f_mass2 = p_mass2 * d2 * lg;
    }
    
    // Glottal mass movement
    // Resultance force on each mass : Sum of the forces arise due to the below conditions
    // 1. Deflection from the equilibrium position (Linear & Non-Linear stiffness of the vocal cord)
    // 2. Contact force when both vocal cords collide (Deformation of each mass)

    // Consider these new parameters (h=Linear stiffness during VF collison & r = damping coefficient)
    // Collison Condition: xi + Ag0i/(2*lg)<=0 then h exist otherwise h=0 (p1237 from IF72)

    float h1_upd = 0.0f;
    float r1_upd = 0.0f;
    float h2_upd = 0.0f;
    float r2_upd = 0.0f;

    // For mass1
    if (x1_curr + Ag01 / (2 * lg) < 0){       
        h1_upd = h1;
        r1_upd = r1_close;
    }
    else{
        h1_upd = 0.0f;
        r1_upd = r1_open;
    }

    // For mass2
    if (x2_curr + Ag02 / (2 * lg) < 0){
        h2_upd = h2;
        r2_upd = r2_close;
    }    
    else{
        h2_upd = 0.0f;
        r2_upd = r2_open;
    }

    //std::cout << "r1_open: " << r1_open << ", r1_close: " << r1_close << std::endl;
    //std::cout << "r2_open: " << r2_open << ", r2_close: " << r2_close << std::endl;
    //exit(EXIT_FAILURE);

    // To calculate the matrix elements use (A-9) from S87 : a11, a12, a21, a22
    // These matrix elemets will be used for furthur vocal cord displacements: x1_next, x2_next for mass m1 and m2
    // Follow (A-10) solution for the code implementation

    a11 = (k1 + h1_upd + kc) / (srate*srate) + r1_upd / srate + m1;
	a12 = -kc / (srate*srate);
	a21 = a12; 
	a22 = (k2 + h2_upd + kc) / (srate*srate) + r2_upd / srate + m2;

    // Follow (A-14) from S87
    s1_prime = k1*etak1*x1_curr*x1_curr*x1_curr + h1_upd*(Ag01 / (2 * lg) + etah1*(Ag01 / (2* lg) + x1_curr)*(Ag01 / (2 * lg) + x1_curr)*(Ag01 / (2 * lg) + x1_curr)); 
	s2_prime = k2*etak2*x2_curr*x2_curr*x2_curr + h2_upd*(Ag02 / (2 * lg) + etah2*(Ag02 / (2* lg) + x2_curr)*(Ag02 / (2 * lg) + x2_curr)*(Ag02 / (2 * lg) + x2_curr));
    
    // Follow (A-11) from S87 to calculate b1 and b2
    // NOTE: You'll find a mismatch between the code implementation and equation
	// that has been provided in the S87 paper in the last term. To understand 
	// the error please validate the units for each terms (which should be: kg*m -in SI) 
    b1 = (2 * m1 + r1_upd / srate)*x1_curr - m1 * x1_old - s1_prime / (srate*srate) + f_mass1/(srate*srate);
	b2 = (2 * m2 + r2_upd / srate)*x2_curr - m2 * x2_old - s2_prime / (srate*srate) + f_mass2/(srate*srate);

    // Compute the determinant 
    float det = a11*a22 - a21*a12;
    if (det == 0)
        det = 1;

    x1_next = (a22*b1 - a12 * b2) / det;
	x2_next = (a11*b2 - a21 * b1) / det;

    // Update x1_curr and x1_old
    x1_old = x1_curr;
	x1_curr = x1_next;
    x1_next = x1_next;
 
	x2_old = x2_curr;
	x2_curr = x2_next;
    x2_next = x2_next;

    // Calculate new area from x1 nad x2
	Ag1 = Ag01 + 2 * lg * x1_curr;
	Ag2 = Ag02 + 2 * lg * x2_curr;

    //*********************GLOTTAL FLOW VELOCITY Ug****************************//
    
    if (Ag1 < 0 || Ag2 < 0){
       ug_next = 0;
    }
    else{
       // Using Equation 3-4 from S87(p957)
       float Rtot = (rho / 2)*fabs(ug_curr)*((0.37 / (Ag1*Ag1)) + (1 - 2 * (Ag2 / A1)*(1 - Ag2 / A1)) / (Ag2*Ag2))  + (12 * mu* lg* lg*(d1 / (Ag1*Ag1*Ag1) + d2 / (Ag2*Ag2*Ag2)));
	   float Ltot = rho * (d1 / Ag1 + d2 / Ag2);
        
       // Equation 2 : Png Noise pressure source has not been considered here
	   ug_next = ((ps - p1) / srate + Ltot * ug_curr) / (Rtot / srate + Ltot);
    }
    
	ug_old = ug_curr;
	ug_curr = ug_next;
    ug_next = ug_next;
}