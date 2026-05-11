#ifndef PARSE_ARGS_H_INCLUDED
#define PARSE_ARGS_H_INCLUDED

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <argp.h>
#include <string.h>

#include "ofuncs.h"


/**
 * @struct SSOConfig
 * @brief Configuration parameters for the SSO algorithm.
 */
struct SSOConfig {
    // population size
    // Default = 100
    size_t np;

    // decision variables
    // Default = 30
    size_t nd;

    // maximum stages
    // Default = 1000
    size_t k_max;

    // number of points in the local search of each stage
    // Default = 10
    size_t rotations;

    // gradient scaling factor.
    // Default = 0.9
    double eta;

    // inertia coefficient or momentum rate.
    // Default = 0.1
    double alpha;

    // velocity limiter for stage k
    // Default = 4
    double beta;

    // objective function to optimize
    // Default = OBJ_RASTRIGIN
    ObjectiveFunction obj;

    uint64_t seed; /**< The seed to use for pseudorandom generation. */

    void *extension; /**< Extension attributes. */
};

error_t parse_args(int argc, char *argv[], struct SSOConfig *config);

/**
 * @brief Parse args allowing for additional options.
 */
error_t parse_args_extend(int argc, char *argv[], struct SSOConfig *config,
                          const struct argp *extension, void *ext_state);

#endif /* PARSE_ARGS_H_INCLUDED */
