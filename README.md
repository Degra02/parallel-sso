# Parallel Shark Smell Optimization (SSO)

This repository contains C implementations of the Shark Smell Optimization (SSO) algorithm with serial and parallel variants (OpenMP, MPI, and hybrid) and simple benchmarking utilities.

## Contents
- Multiple `main_*` programs at the repository root implementing serial, OpenMP, MPI and hybrid variants.
- `src/sso/`: core SSO implementation, argument parsing and utilities.
- `benchmarks/`: plotting script and raw results (plots are written to `benchmarks/plots`).
- `tests/`: scripted test/launch helpers.
- `Makefile`: top-level build rules.

## Build
From the repository root, run `make` to build available targets defined in the `Makefile`.

In order to build specific variants, the `Makefile` expects the following pattern:
- `make sso_<parallelism>_<variant>`, where `<parallelism>` is one of `openmp`, `mpi`, or `hybrid`, and `<variant>` is a specific parallelization strategy, i.e. `sharks`, `dim` or `hybrid`.

Example:
- `make sso_openmp_sharks` builds the OpenMP version parallelized over sharks

== Run
```txt
Usage: sso_<parallelism>_<variant> [OPTION...]
Program to perform shark smelling optimization.

  -d, --nd=int               The number of decision variables.
  -k, --k_max=int            The number of stages.
  -p, --np=int               The population size.
  -t, --threads=int          The population size. (only for OpenMP/hybrid)
  -a, --alpha=[0-1]          Momentum (inertia) rate.
  -b, --beta=[0-1]           Velocity limiter ratio.
  -m, --rotations=int        Rotational points to check at each step.
  -n, --eta=[0-1]            Gradient scaling factor.
  -o, --obj=int              The objective function.
  -s, --seed=int             PRNG seed (0 for random).
  -?, --help                 Give this help list
      --usage                Give a short usage message
```


Tests
- Use `tests/launch_tests.sh` to run the provided test scenarios.