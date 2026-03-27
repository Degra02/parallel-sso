#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

/// @brief  Objective functions (taken from the paper)
typedef enum {
  OBJ_RASTRIGIN = 0,
  OBJ_GRIEWANGK = 1,
  OBJ_SCHAFFER = 2,
  /// TODO: Add more objective functions
} ObjectiveFunction;

/// @brief  Configuration for the SSO algorithm
typedef struct {
  // population size
  uint32_t np;

  // decision variables
  uint32_t nd;

  // maximum stages
  uint32_t kmax;

  // number of points in the local search of each stage
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

  ObjectiveFunction obj;
  uint64_t seed;
} SSOConfig;

#endif
