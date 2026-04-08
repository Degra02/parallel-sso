#ifndef OFUNCS_H
#define OFUNCS_H

#include <stdint.h>
#include <math.h>
#include <stddef.h>

/// @brief  Objective functions (taken from the paper)
typedef enum {
  OBJ_RASTRIGIN = 0,
  OBJ_GRIEWANGK = 1,
  OBJ_SCHAFFER = 2,
  /// TODO: Add more objective functions
} ObjectiveFunction;

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

double domain_lb(ObjectiveFunction obj);
double domain_ub(ObjectiveFunction obj);

// Benchmark object functions

// Rastrigin eq 12:
double rastrigin(const double *x, uint32_t nd);

//Griegwangk eq 13:
double griewangk(const double *x, uint32_t nd);

// Schaffer eq 14:
double schaffer(const double *x, uint32_t nd);

// 3 OF dispatcher 
double eval_min(const double *x, uint32_t nd, ObjectiveFunction obj);

inline double OF(const double *x, uint32_t nd, ObjectiveFunction obj) {
    return -eval_min(x, nd, obj);
}

#endif /* OFUNCS_H */