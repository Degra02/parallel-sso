#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <stdlib.h>
#include "types.h"

/**
 * @brief Generate a (pseudo)random value uniformly distributed in the given interval.
 */
static inline double utils_rand(double min, double max) {
    return (double) rand() / ((double) RAND_MAX + 1.0) * (max - min) + min;
}

/**
 * @def CLAMP(x, interval)
 * @brief Clamp the given value x using interval as bounds.
 * @param x The value to clamp.
 * @param interval The clamping interval.
 */
static inline double utils_clamp(double x, const struct Interval *interval) {
    if (x < interval->start) return interval->start;
    if (x > interval->end) return interval->end;
    return x;
}

//Clamp every dimension of x into [lb[j], ub[j]]. */
static inline void utils_clamp_vec(double *x, size_t num_dim,
                                   const struct Interval *domain) {
    for (size_t dim = 0; dim < num_dim; dim++) {
        x[dim] = utils_clamp(x[dim], &domain[dim]);
    }
}

#endif /* UTILS_H_INCLUDED */
