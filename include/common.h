#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

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
  uint32_t k_max;

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


typedef struct {
    const char *name;
    ObjectiveFunction value;
} ObjEntry;

static const ObjEntry obj_registry[] = {
    {"rastrigin", OBJ_RASTRIGIN},
    {"griewangk",  OBJ_GRIEWANGK},
    {"schaffer",   OBJ_SCHAFFER},
    {NULL, 0}
};

static struct option long_opts[] = {
    {"np",    required_argument, 0, 1},
    {"nd",    required_argument, 0, 2},
    {"k_max",  required_argument, 0, 3},
    {"m",     required_argument, 0, 4},
    {"n_k",   required_argument, 0, 5},
    {"a_k",   required_argument, 0, 6},
    {"b_k",   required_argument, 0, 7},
    {"seed",  required_argument, 0, 8},
    {"obj", required_argument, 0, 9},
    {0, 0, 0, 0}
};

ObjectiveFunction lookup_obj(const char *name);
void parse_args(int argc, char **argv, SSOConfig *cfg);

#endif
