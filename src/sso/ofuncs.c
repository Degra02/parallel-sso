#include "ofuncs.h"
#include "types.h"
#include <stdlib.h>

// 1 Domain bounds
static double domain_lowerbound(ObjectiveFunction obj, [[maybe_unused]] size_t dim) {
    switch (obj) {
        case OBJ_RASTRIGIN: return -20.0;   /* paper Eq. 12: -20 ≤ xi ≤ 20   */
        case OBJ_GRIEWANGK: return -600.0;  /* paper Eq. 13: -600 ≤ xk ≤ 600 */
        case OBJ_SCHAFFER:  return -100.0;  /* paper Eq. 14: -100 ≤ xi ≤ 100 */
        default:            return -100.0;
    }
}

static double domain_upperbound(ObjectiveFunction obj, [[maybe_unused]] size_t dim) {
    switch (obj) {
        case OBJ_RASTRIGIN: return  20.0;
        case OBJ_GRIEWANGK: return  600.0;
        case OBJ_SCHAFFER:  return  100.0;
        default:            return  100.0;
    }
}

struct Interval *obj_alloc_domain_bounds(ObjectiveFunction obj, size_t dim_num) {
    struct Interval *domain = calloc(dim_num, sizeof(struct Interval));
    if (domain == NULL) return NULL;

    for (size_t j = 0; j < dim_num; ++j) {
        domain[j] = (struct Interval) {
            .start = domain_lowerbound(obj, j),
            .end = domain_upperbound(obj, j)
        };
    }

    return domain;
}


//2  Benchmark object functions

// Rastrigin eq 12:
static double rastrigin(const double *x, uint32_t nd) {
    const double A = 10.0;
    double val = A * (double)nd;
    for (uint32_t i = 0; i < nd; i++)
        val += x[i] * x[i] - A * cos(2.0 * M_PI * x[i]);
    return val;
}

static double rastrigin_derivative(const double *x, [[maybe_unused]] size_t nd, size_t dim) {
    const double A = 10.0;
    double val = 2 * x[dim] + A * 2.0 * M_PI * sin(2.0 * M_PI * x[dim]);
    return val;
}

//Griegwangk eq 13:
static double griewangk(const double *x, uint32_t nd) {
    double sum = 0.0, prod = 1.0;
    for (uint32_t k = 0; k < nd; k++) {
        sum  += x[k] * x[k] / 4000.0;
        prod *= cos(x[k] / sqrt((double)(k + 1)));
    }
    return sum - prod + 1.0;
}

static double griewangk_derivative(const double *x, size_t nd, size_t dim) {
    double prod = 1.0;
    for (size_t k = 0; k < nd; ++k) {
        double root = sqrt((double) (k + 1));
        if (k != dim) {
            prod *= cos(x[k] / root);
        } else {
            prod *= sin(x[k] / root) / root;
        }
    }
    return x[dim] / 2000.0 + prod;
}

// Schaffer eq 14:
static double schaffer(const double *x, uint32_t nd) {
    // TODO: Does this need to be generalized to more dimensions?
    (void)nd;
    double r2   = x[0]*x[0] + x[1]*x[1];
    double sin2 = sin(sqrt(r2));
    sin2 *= sin2;
    double denom = 1.0 + 0.001 * r2;
    denom *= denom;
    return 0.5 + (sin2 - 0.5) / denom;
}

static double schaffer_derivative(const double *x, [[maybe_unused]] size_t nd, size_t dim) {
    if (dim >= 2) return 0.0;
    // TODO:
    return 0.0;
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

double eval_derivative(const double *x, size_t nd, ObjectiveFunction obj, size_t dim) {
    switch (obj) {
        case OBJ_RASTRIGIN: return rastrigin_derivative(x, nd, dim);
        case OBJ_GRIEWANGK: return griewangk_derivative(x, nd, dim);
        case OBJ_SCHAFFER:  return schaffer_derivative(x, nd, dim);
        default:            return 0.0;
    }
}
