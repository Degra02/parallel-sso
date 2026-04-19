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
 * @brief Clamp the given value x using interval as bound.
 * @param x The value to clamp.
 * @param interval The clamping interval.
 * @return x if x is contained in interval, otherwise the nearest boundary.
 * @pre interval start <= end
 */
static inline double utils_clamp(double x, const struct Interval *interval) {
    if (x < interval->start) return interval->start;
    if (x > interval->end) return interval->end;
    return x;
}

/**
 * @brief Clamp every value in x with the boundaries contained in domain.
 * @param x The vector to clamp.
 * @param num_dim The size of the vector.
 * @param domain An interval boundary for every dimension.
 * @pre interval start <= end for every interval in domain.
 */
static inline void utils_clamp_vec(double x[], size_t num_dim,
                                   const struct Interval domain[]) {
    for (size_t dim = 0; dim < num_dim; dim++) {
        x[dim] = utils_clamp(x[dim], &domain[dim]);
    }
}

#endif /* UTILS_H_INCLUDED */
