#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "sso/parse_args.h"
#include "sso/sso.h"
#include "sso/ofuncs.h"

static void print_result(double best_min, const double *best_pos, size_t nd) {
  printf("\n=== Final Result ===\n");
  printf("Best f(x) = %.10e\n", best_min);

  size_t count_per_row = 8;
  printf("Best x    = [");
  for (uint32_t j = 0; j < nd; j++) {
    if (j % count_per_row == 0)
      printf("\n");
    printf(" %12.8f", best_pos[j]);
  }
  printf("\n]\n");
}

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

    // reduction result shared across ranks
    double global_best_min = INFINITY;

    // Stage loop
    for (size_t k = 0; k < cfg.k_max; ++k) {
        // Perform local step on this rank's sharks
        sso_perform_step(sharks, &cfg_local, domain, scratch, &local_best_min,
             local_best_pos);

        // Prepare pair for MPI_MINLOC: (value, rank)
        struct { double val; int rank; } local_pair, global_pair;
        local_pair.val = local_best_min;
        local_pair.rank = rank;

        MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        global_best_min = global_pair.val;

        // Keep the local state aligned with the reduced best so the next stage
        // starts from the same global threshold on every rank.
        local_best_min = global_best_min;

        if (rank == global_pair.rank) {
          memcpy(global_best_pos, local_best_pos, cfg.nd * sizeof(double));
        }

        // Broadcast the best position from the winner rank
        MPI_Bcast(global_best_pos, (int)cfg.nd, MPI_DOUBLE, global_pair.rank,
            MPI_COMM_WORLD);

        memcpy(local_best_pos, global_best_pos, cfg.nd * sizeof(double));

        // Progress report only on rank 0
        if (rank == 0 && (k == 0 || (k + 1) % 100 == 0)) {
            printf("Stage %5lu | best f(x) = %.8e\n", k + 1, global_best_min);
        }
    }

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
