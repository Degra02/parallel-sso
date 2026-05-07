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

/**
 * @brief Print the configuration parameters at the start of the execution.
 * @param cfg The configuration struct to print.
 */
static void print_info(const struct SSOConfig *cfg, char *variant_name) {
    printf("=== SSO %s === \n", variant_name);
    printf("\n NP=%lu\t\tND=%lu\t\tk_max=%lu\tM=%lu\n",
           cfg->np, cfg->nd, cfg->k_max, cfg->rotations);
    printf("eta=%.3f\talpha=%.3f\tbeta=%.3f\n",
           cfg->eta, cfg->alpha, cfg->beta);
    printf("objective=%d\tseed=%lu\n\n", cfg->obj, cfg->seed);
}

/**
 * @brief Print the final result at the end of the execution.
 * @param best_min The best minimum value found.
 * @param best_pos The position of the best minimum value found.
 * @param nd The number of dimensions of the problem.
 */
static void print_result(double best_min, const double *best_pos, size_t nd) {
    printf("\n=== Final Result ===\n");
    printf("Best f(x) = %.10e\n", best_min);

    size_t count_per_row = 8;
    printf("Best x    = [");
    for (uint32_t j = 0; j < nd; j++) {
        if (j % count_per_row == 0) printf("\n");
        printf(" %9.6f", best_pos[j]);
    }
    printf("\n]\n");
}

#endif /* UTILS_H_INCLUDED */
