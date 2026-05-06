#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "sso/parse_args.h"
#include "sso/sso.h"
#include "sso/ofuncs.h"
#include "sso/utils.h"
#include "common/utils.h"


/**
 * @brief MPI parallel sharks algorithm entrypoint.
 */
int main(int argc, char *argv[]) {
    int provided = 0;
    // MPI_THREAD_FUNNELED indicates that if the process is multithreaded, only the thread that called MPI_Init_thread will make MPI calls.
    // TODO: check if it's useful with our use case [docs](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Init_thread.3.html)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct SSOConfig cfg;
    if (parse_args(argc, argv, &cfg) != 0) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        print_info(&cfg);
    }

    // Compute local population size (block distribution)
    size_t base = cfg.np / (size_t)size;
    size_t rem = cfg.np % (size_t)size;
    size_t local_np = base + (rank < (int)rem ? 1 : 0);

    // Seed PRNG differently per rank to get different initial populations
    unsigned seed = (unsigned)(cfg.seed == 0 ? (unsigned)time(NULL) : (unsigned)cfg.seed);
    srand(seed + rank);

    // Domain bounds (replicated on every rank)
    struct Interval *domain = obj_alloc_domain_bounds(cfg.obj, cfg.nd);

    // Per-rank config copy with adjusted population
    struct SSOConfig cfg_local = cfg;
    cfg_local.np = local_np;

    // Allocate local sharks
    struct Shark *sharks = sso_sharks_alloc(domain, &cfg_local);
    double *scratch = calloc(cfg.nd, sizeof(double));
    double *local_best_pos = calloc(cfg.nd, sizeof(double));
    double *global_best_pos = calloc(cfg.nd, sizeof(double));

    if (domain == NULL || sharks == NULL || scratch == NULL ||
      local_best_pos == NULL || global_best_pos == NULL) {
        if (rank == 0) perror("Malloc error");
        if (sharks) sso_sharks_free(sharks, cfg_local.np);
        free(domain);
        free(scratch);
        free(local_best_pos);
        free(global_best_pos);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // minimisation value tracked per rank
    double local_best_min = INFINITY;

    BENCHMARK(total_start, "Total MPI time") {
        // Stage loop
        for (size_t k = 0; k < cfg.k_max; ++k) {
            // Perform local step on this rank's sharks
            sso_perform_step(sharks, &cfg_local, domain, scratch, &local_best_min,
                local_best_pos);
        }
    }

    // Barrier to ensure all ranks have finished their local search before reduction.
    MPI_Barrier(MPI_COMM_WORLD);
    
    double global_best_min = INFINITY;
    MPI_Allreduce(&local_best_min, &global_best_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    int best_rank = -1;
    if (local_best_min == global_best_min) {
        best_rank = rank;
    }
    MPI_Allreduce(MPI_IN_PLACE, &best_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    // The rank with the best solution copies to global_best_pos before broadcast
    if (best_rank == rank) {
        memcpy(global_best_pos, local_best_pos, cfg.nd * sizeof(double));
    }
    
    // All ranks broadcast to get the global best position
    MPI_Bcast(global_best_pos, (int)cfg.nd, MPI_DOUBLE, best_rank, MPI_COMM_WORLD);

    if (rank == 0) {
        print_result(global_best_min, global_best_pos, cfg.nd);
    }

    // Cleanup
    free(global_best_pos);
    free(local_best_pos);
    free(scratch);
    sso_sharks_free(sharks, cfg_local.np);
    free(domain);

    MPI_Finalize();
    return 0;
}
