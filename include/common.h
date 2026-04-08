#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include "ofuncs.h"


/// @brief  Configuration for the SSO algorithm
typedef struct {
  // population size
  // Default = 100
  uint32_t np;

  // decision variables
  // Default = 30
  uint32_t nd;

  // maximum stages
  // Default = 1000
  uint32_t k_max;

  // number of points in the local search of each stage
  // Default = 10
  uint32_t m;

  // gradient scaling factor. 
  // Default = 0.9
  double n_k;

  // inertia coefficient or momentum rate.
  // Default = 0.1
  double a_k;

  // velocity limiter for stage k
  // Default = 4
  double b_k;

  // objective function to optimize
  // Default = OBJ_RASTRIGIN
  ObjectiveFunction obj;

  uint64_t seed;
} SSOConfig;

ObjectiveFunction lookup_obj(const char *name);
void parse_args(int argc, char **argv, SSOConfig *cfg);

#endif
