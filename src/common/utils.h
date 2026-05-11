#ifndef COMMON_UTILS_H_INCLUDED
#define COMMON_UTILS_H_INCLUDED

#include <mpi.h>
#include <stdio.h>

#define MAIN_RANK 0
#define IF_MAIN_PROC if (rank == MAIN_RANK)
#define BENCHMARK(start, prompt)\
            for (double start = MPI_Wtime(); start != -1.0F;\
                print_time(prompt ": %lf s\n", start, rank), start = -1.0F)
#define BENCHMARK_SERIAL(start, prompt)\
            for (double start = MPI_Wtime(); start != -1.0F;\
                print_time(prompt ": %lf s\n", start, MAIN_RANK), start = -1.0F)
#define BENCHMARK_OPENMP(start, prompt) \
            for (double start = omp_get_wtime(); start != -1.0F; \
                printf(prompt ": %lf s\n", omp_get_wtime() - start), start = -1.0F)

static inline void print_time(const char* prompt, double start, int rank) {
    IF_MAIN_PROC {
        printf(prompt, MPI_Wtime() - start);
    }
}

#endif /* COMMON_UTILS_H_INCLUDED */
