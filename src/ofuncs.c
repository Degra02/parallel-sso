#include "ofuncs.h"

// 1 Domain bounds
double domain_lb(ObjectiveFunction obj) {
    switch (obj) {
        case OBJ_RASTRIGIN: return -20.0;   /* paper Eq. 12: -20 ≤ xi ≤ 20   */
        case OBJ_GRIEWANGK: return -600.0;  /* paper Eq. 13: -600 ≤ xk ≤ 600 */
        case OBJ_SCHAFFER:  return -100.0;  /* paper Eq. 14: -100 ≤ xi ≤ 100 */
        default:            return -100.0;
    }
}

double domain_ub(ObjectiveFunction obj) {
    switch (obj) {
        case OBJ_RASTRIGIN: return  20.0;
        case OBJ_GRIEWANGK: return  600.0;
        case OBJ_SCHAFFER:  return  100.0;
        default:            return  100.0;
    }
}


//2  Benchmark object functions

// Rastrigin eq 12:
double rastrigin(const double *x, uint32_t nd) {
    const double A = 10.0;
    double val = A * (double)nd;
    for (uint32_t i = 0; i < nd; i++)
        val += x[i] * x[i] - A * cos(2.0 * M_PI * x[i]);
    return val;
}

//Griegwangk eq 13:

double griewangk(const double *x, uint32_t nd) {
    double sum = 0.0, prod = 1.0;
    for (uint32_t k = 0; k < nd; k++) {
        sum  += x[k] * x[k] / 4000.0;
        prod *= cos(x[k] / sqrt((double)(k + 1)));
    }
    return sum - prod + 1.0;
}

// Schaffer eq 14:

double schaffer(const double *x, uint32_t nd) {
    (void)nd;
    double r2   = x[0]*x[0] + x[1]*x[1];
    double sin2 = sin(sqrt(r2));
    sin2 *= sin2;
    double denom = 1.0 + 0.001 * r2;
    denom *= denom;
    return 0.5 + (sin2 - 0.5) / denom;
}

// 3 OF dispatcher 

double eval_min(const double *x, uint32_t nd, ObjectiveFunction obj) {
    switch (obj) {
        case OBJ_RASTRIGIN: return rastrigin(x, nd);
        case OBJ_GRIEWANGK: return griewangk(x, nd);
        case OBJ_SCHAFFER:  return schaffer(x, nd);
        default:            return 0.0;
    }
}