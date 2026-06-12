# Parallel Shark Smell Optimization

This project implements and evaluates parallel versions of the Shark Smell
Optimization (SSO) algorithm. SSO is a population-based metaheuristic inspired
by the way sharks locate prey by following odor gradients.

The repository contains:

- a sequential implementation in C;
- shared-memory implementations using OpenMP;
- distributed-memory implementations using MPI;
- hybrid MPI + OpenMP implementations;
- scripts and raw data for strong- and weak-scaling experiments;
- Python utilities for generating performance plots.

The implementations are compared through execution time, speedup, efficiency,
and scalability on the University of Trento HPC cluster.

- **Course:** High Performance Computing for Data Science
- **Institution:** University of Trento, DISI
- **Academic year:** 2025-2026

## Parallelization Strategies

The project investigates different ways of exposing parallelism in the SSO
algorithm:

| Strategy | Description |
| --- | --- |
| Shark-level | Distributes independent sharks across workers. |
| Dimension-level | Parallelizes position and velocity updates across search-space dimensions. |
| Rotation-level | Distributes the probes used during the rotational local search. |
| Combined | Uses more than one decomposition level within the same implementation. |

These strategies are implemented with OpenMP, MPI, and hybrid MPI + OpenMP
programming models. The hybrid versions combine inter-node distribution with
intra-node shared-memory parallelism.

The available executables follow this naming convention:

```text
sso_<programming_model>_<strategy>
```

Examples include `sso_openmp_sharks`, `sso_mpi_dim`, and
`sso_hybrid_sharks`.

## Requirements

### Core software

- a C compiler with OpenMP support;
- an MPI implementation such as OpenMPI or MPICH;
- GNU Make;
- a Unix-like shell.

### Benchmarking and plotting

- Python 3;
- Matplotlib;
- a PBS-compatible scheduler for the provided cluster submission scripts.

The project has been developed and benchmarked with GCC and OpenMPI on the
University of Trento HPC cluster.

## Building

Clone the repository:

```bash
git clone https://github.com/Degra02/parallel-sso.git
cd parallel-sso
```

Build the sequential implementation:

```bash
make sso_serial
```

Build a specific parallel implementation using its programming model and
strategy:

```bash
make sso_openmp_sharks
make sso_mpi_sharks
make sso_hybrid_sharks
```

Other available strategies include `dim`, `rot`, and `hybrid`, depending on
the selected programming model. Run `make help` to display the build help and
`make clean` to remove generated objects and executables.

## Running

### Sequential

```bash
./sso_serial -p 1000 -d 200 -k 1000 -m 50
```

### OpenMP

```bash
./sso_openmp_sharks -p 1000 -d 200 -k 1000 -m 50 -t 8
```

### MPI

```bash
mpirun -np 8 ./sso_mpi_sharks -p 1000 -d 200 -k 1000 -m 50
```

### Hybrid MPI + OpenMP

```bash
export OMP_NUM_THREADS=8
mpirun -np 4 ./sso_hybrid_sharks \
  -p 1000 -d 200 -k 1000 -m 50 -t 8
```

The most relevant command-line options are:

| Option | Description |
| --- | --- |
| `-p`, `--np` | Population size |
| `-d`, `--nd` | Number of decision variables |
| `-k`, `--k_max` | Number of optimization stages |
| `-m`, `--rotations` | Number of rotational-search probes |
| `-t`, `--threads` | Number of OpenMP threads, where supported |
| `-o`, `--obj` | Objective function: `rastrigin`, `griewangk`, or `schaffer` |
| `-s`, `--seed` | Random seed; use `0` for a time-based seed |

Use the `--help` option on an executable to see its complete argument list.

## Benchmarking

The repository includes PBS templates and launch scripts for running the
implementations with different process and thread counts.

For example, generate and submit an MPI shark-level job with:

```bash
bash tests/launch_tests.sh \
  --exec tests/mpi_sharks \
  --job mpi_sharks \
  --procs 8 | qsub
```

The shark-level weak-scaling experiments have dedicated launchers:

```bash
bash tests/weak_scaling_sharks/launch_weak_scaling_sharks.sh
bash tests/weak_scaling_sharks/launch_weak_scaling_hybrid_sharks.sh
```

These scripts are intended to be executed on a configured PBS cluster. Raw
measurements are stored under `benchmarks/raw/`.

Generate the performance plots with:

```bash
python3 benchmarks/plot_results.py
```

The generated figures are written to `benchmarks/plots/`.

## Project Structure

```text
parallel-sso/
|-- Makefile                    Build rules
|-- README.md
|-- src/
|   |-- main_serial.c           Sequential entry point
|   |-- main_openmp_*.c         OpenMP implementations
|   |-- main_mpi_*.c            MPI implementations
|   |-- main_hybrid_*.c         MPI + OpenMP implementations
|   |-- common/                 Shared timing utilities
|   `-- sso/                    Core algorithm and objective functions
|-- tests/                      PBS templates and launch scripts
|-- benchmarks/
|   |-- raw/                    Collected measurements
|   |-- plots/                  Generated figures
|   `-- plot_results.py         Plotting utility
`-- docs/
    `-- report/                 Project report and references
```

## Objective Functions

The implementation supports three benchmark functions from the original SSO
work:

- Rastrigin;
- Griewangk;
- Schaffer.

Rastrigin is the default objective function used in the performance
experiments.

## Authors

- Pietro Cipriani
- Filippo De Grandi
- Giacomo Vettore
