#ifndef COMMON_UTILS_H_INCLUDED
#define COMMON_UTILS_H_INCLUDED

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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

static inline void free_matrix(size_t n, void *matrix) {
    for (size_t i = 0; i < n; ++i) {
        free(((void**)matrix)[i]);
    }

    free(matrix);
}

static inline void *calloc_matrix(size_t n, size_t m, size_t size) {
    void **matrix = (void**) calloc(n, sizeof(void*));
    if (matrix == NULL) return NULL;

    for (size_t i = 0; i < n; ++i) {
        matrix[i] = calloc(m, size);
        if (matrix[i] == NULL) {
            free_matrix(n, matrix);
            return NULL;
        }
    }

    return matrix;
}

static inline double thread_rand_r(unsigned int *seedp, double min, double max) {
    return (double) rand_r(seedp) / ((double) RAND_MAX + 1.0) * (max - min) + min;
}

#endif /* COMMON_UTILS_H_INCLUDED */
