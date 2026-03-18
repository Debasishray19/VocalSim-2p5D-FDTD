/*
 * SimulationUnits.h
 *
 * Define SI-based units
 *
 */

#ifndef SIMULATIONUNITS_H_
#define SIMULATIONUNITS_H_
 
const float METER = 1;
const float CENTIMETER = 1e-2 * METER;
const float MILLIMETER = 1e-3 * METER;
 
const float SECOND = 1;
const float MILLISECOND = 1e-3 * SECOND;
 
const float HERTZ = 1 / SECOND;
const float KILOHERTZ = 1e3 * HERTZ;
 
const float KILOGRAM = 1;
const float GRAM = 1e-3 * KILOGRAM;

const float NEWTON = KILOGRAM * METER / (SECOND * SECOND);
const float DYN    = 1e-5 * NEWTON; 
 
#endif // SIMULATIONUNITS_H_