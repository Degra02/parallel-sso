/*
* Algorithm overview:
 *  1. Initialise NP sharks at random positions in the feasible domain.
 *  2. For each stage k = 1 … k_max:
 *       For each shark i = 1 … NP:
 *         a. Compute numerical gradient ∇OF at X^k_i            (Eq. 6)
 *         b. Update velocity with momentum + limiter             (Eq. 7-8)
 *         c. Forward step  Y^{k+1}_i = X^k_i + V^k_i · Δt      (Eq. 9)
 *         d. Rotational local search: M probe points around Y    (Eq. 10)
 *         e. Select best candidate as X^{k+1}_i                 (Eq. 11)
 *  3. Return the best individual found across all stages.
 */


//to choose the number of processes assigned to dimension parallelism: mpirun -np 8 ./sso_mpi_hybrid --dim-procs=4

#include "sso/parse_args.h"
#include "sso/ofuncs.h"
#include "sso/sso.h"
#include "sso/utils.h"
#include "common/utils.h"

#include <errno.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>

enum Msgs {
    MSG_POS = 1
};

struct Args {
    size_t dim_procs; /**< MPI processes used for dimension-level parallelism. */
};

static error_t parser(int key, char *arg, struct argp_state *state);

static const struct argp_option options[] = {
    {"dim-procs", 'D', "int", 0,
     "MPI processes used for dimension-level parallelism; 0 keeps the default.", 1},
    { 0 }
};

static const struct argp argp = {options, parser, "", NULL, 0, 0, 0};

/**
 * @brief MPI hybrid algorithm entrypoint.
 */
int main(int argc, char *argv[]) {
    // TODO: argp allows to combine different parsers. We should be able to
    // define serial/parallel specific arguments without rewriting everything.
    struct SSOConfig cfg;
    struct Args args = {0};
    if (parse_args_extend(argc, argv, &cfg, &argp, &args) != 0) {
        return EXIT_FAILURE;
    }

    if (0 != MPI_Init(&argc, &argv)) return EXIT_FAILURE;

    int size, rank;
    if (0 != MPI_Comm_size(MPI_COMM_WORLD, &size)) return EXIT_FAILURE;
    if (0 != MPI_Comm_rank(MPI_COMM_WORLD, &rank)) return EXIT_FAILURE;

    size_t requested_dim_procs = args.dim_procs;
    size_t dim_procs_size = requested_dim_procs != 0
                                ? requested_dim_procs
                                : (size > 1 ? 2u : 1u);

    if (dim_procs_size == 0 || dim_procs_size > (size_t) size) {
        IF_MAIN_PROC {
            fprintf(stderr,
                    "dim-procs must be between 1 and the total number of MPI processes (%d)\n",
                    size);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int dim_procs = (int) dim_procs_size;
    if (size % dim_procs != 0) {
        IF_MAIN_PROC {
            fprintf(stderr, "Number of MPI processes must be divisible by %d\n",
                    dim_procs);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int shark_blocks = size / dim_procs;
    int shark_block = rank / dim_procs;
    int dim_part = rank % dim_procs;

    MPI_Comm dim_comm;
    MPI_Comm_split(MPI_COMM_WORLD, shark_block, dim_part, &dim_comm);

    int dim_size, dim_rank;
    MPI_Comm_size(dim_comm, &dim_size);
    MPI_Comm_rank(dim_comm, &dim_rank);

    // Seed the PRNG to have reproducible runs. 0 for time-based randomness.
    unsigned init_seed = cfg.seed == 0 ? (unsigned) time(NULL)
                                       : (unsigned) cfg.seed;
    MPI_Bcast(&init_seed, 1, MPI_UNSIGNED, MAIN_RANK, MPI_COMM_WORLD);

    unsigned seed = init_seed + ((unsigned) shark_block << 16);
    srand(seed);

    IF_MAIN_PROC {
        print_info(&cfg, "MPI Hybrid");
    }

    // Compute local population size.
    size_t base = cfg.np / (size_t) shark_blocks;
    size_t rem = cfg.np % (size_t) shark_blocks;
    size_t local_np = base + (shark_block < (int) rem ? 1 : 0);

    // Domain bounds.
    struct Interval *domain = obj_alloc_domain_bounds(cfg.obj, cfg.nd);

    // Per-block config copy with adjusted population.
    struct SSOConfig local_cfg = cfg;
    local_cfg.np = local_np;

    // Alloc local population. Processes in the same block initialise the same sharks.
    struct Shark *local_sharks = sso_sharks_alloc(domain, &local_cfg);

    // Scratch array to avoid allocations in loops.
    double *scratch = calloc(cfg.nd, sizeof(double));

    // The best position found up to date.
    double *local_best_pos = calloc(cfg.nd, sizeof(double));
    double *global_best_pos = calloc(cfg.nd, sizeof(double));

    int *dim_sizes = calloc((size_t) dim_size, sizeof(int));
    int *dim_starts = calloc((size_t) dim_size, sizeof(int));

    int ret;
    if (domain == NULL || local_sharks == NULL || scratch == NULL
        || local_best_pos == NULL || global_best_pos == NULL
        || dim_sizes == NULL || dim_starts == NULL) {
        perror("Malloc error");
        ret = EXIT_FAILURE;
    } else {
        // Local best. Minimisation problem.
        double local_best_min = INFINITY;
        double global_best_min = INFINITY;

        size_t start_dim = 0, end_dim = 0;
        for (int i = 0; i < dim_size; ++i) {
            size_t remainder = cfg.nd % (size_t) dim_size;
            dim_starts[i] = (int)(cfg.nd / (size_t) dim_size * (size_t) i
                            + ((size_t) i < remainder ? (size_t) i : remainder));
            dim_sizes[i] = (int)(cfg.nd / (size_t) dim_size
                            + ((size_t) i < remainder ? 1 : 0));

            if (i == dim_rank) {
                start_dim = (size_t) dim_starts[i];
                end_dim = start_dim + (size_t) dim_sizes[i];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        BENCHMARK(total_start, "Total time") {

            for (size_t shark = 0; shark < local_cfg.np; ++shark) {
                struct Shark *shark_ptr = &local_sharks[shark];

                // Perform k_max movement stages.
                for (size_t k = 0; k < local_cfg.k_max; ++k) {
                    double R1;
                    double R2;

                    if (dim_rank == 0) {
                        R1 = utils_rand(0, 1);
                        R2 = utils_rand(0, 1);
                    }

                    MPI_Bcast(&R1, 1, MPI_DOUBLE, 0, dim_comm);
                    MPI_Bcast(&R2, 1, MPI_DOUBLE, 0, dim_comm);

                    // Update shark speed.
                    for (size_t dim = start_dim; dim < end_dim; ++dim) {
                        double v_prev = shark_ptr->speed[dim];

                        double derivative = eval_derivative(shark_ptr->position,
                                                    local_cfg.nd, local_cfg.obj, dim);
                        // gradient based speed: γ·R1·gradient
                        double grad_term = local_cfg.eta * R1 * derivative;
                        // momentum based speed: α·R2·v_{k-1}
                        double mom_term  = local_cfg.alpha * R2 * v_prev;
                        shark_ptr->speed[dim] = grad_term + mom_term;

                        if (fabs(v_prev) >= 1e-15) {
                            // Limit the velocity up to β·v.
                            double limit = local_cfg.beta * v_prev;
                            if (fabs(shark_ptr->speed[dim]) > fabs(limit)) {
                                shark_ptr->speed[dim] = limit;
                            }
                        }
                    }

                    for (size_t dim = start_dim; dim < end_dim; ++dim) {
                        // Update shark position with a forward movement.
                        // Assume Δt = 1.
                        shark_ptr->position[dim] = utils_clamp(
                                shark_ptr->position[dim] + shark_ptr->speed[dim],
                                &domain[dim]);
                    }

                    MPI_Allgatherv(shark_ptr->position + start_dim,
                        (int)(end_dim - start_dim), MPI_DOUBLE,
                        shark_ptr->position, dim_sizes,
                        dim_starts, MPI_DOUBLE,
                        dim_comm);

                    // Current best candidate.
                    double *candidate = scratch;
                    double best = OF(shark_ptr->position,
                                     local_cfg.nd, local_cfg.obj);
                    double best_r3 = 0.0;

                    for (uint32_t m = 0; m < local_cfg.rotations; ++m) {
                        double r3;

                        if (dim_rank == 0) {
                            r3 = utils_rand(-1, 1);
                        }

                        MPI_Bcast(&r3, 1, MPI_DOUBLE, 0, dim_comm);

                        // "Rotate" around the shark position.
                        for (size_t dim = start_dim; dim < end_dim; ++dim) {
                            candidate[dim] = utils_clamp(
                                        shark_ptr->position[dim] * (1 + r3),
                                        &domain[dim]);
                        }

                        MPI_Allgatherv(candidate + start_dim,
                            (int)(end_dim - start_dim), MPI_DOUBLE,
                            candidate, dim_sizes,
                            dim_starts, MPI_DOUBLE,
                            dim_comm);

                        // Update position if better than the current one.
                        double val = OF(candidate, local_cfg.nd, local_cfg.obj);
                        if (val > best) {
                            best = val;
                            best_r3 = r3;
                        }
                    }

                    if (best_r3 != 0.0) {
                        for (size_t dim = start_dim; dim < end_dim; ++dim) {
                            shark_ptr->position[dim] = utils_clamp(
                                        shark_ptr->position[dim] * (1 + best_r3),
                                        &domain[dim]);
                        }

                        MPI_Allgatherv(shark_ptr->position + start_dim,
                            (int)(end_dim - start_dim), MPI_DOUBLE,
                            shark_ptr->position, dim_sizes,
                            dim_starts, MPI_DOUBLE,
                            dim_comm);
                    }

                    // If we have an all-time best value for this block, update it.
                    double cur_min = -best;
                    if (cur_min < local_best_min) {
                        local_best_min = cur_min;
                        memcpy(local_best_pos, shark_ptr->position,
                               local_cfg.nd * sizeof(double));
                    }
                }
            }

            // Ensure everyone has finished.
            MPI_Barrier(MPI_COMM_WORLD);

            struct {
                double value;
                int rank;
            } local_pair, global_pair;

            local_pair.value = local_best_min;
            local_pair.rank = rank;

            MPI_Allreduce(&local_pair, &global_pair, 1,
                          MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

            int best_rank = global_pair.rank;
            if (rank == MAIN_RANK) {
                global_best_min = global_pair.value;
            }

            if (rank == best_rank) {
                if (rank == MAIN_RANK) {
                    memcpy(global_best_pos, local_best_pos,
                           cfg.nd * sizeof(double));
                } else {
                    MPI_Send(local_best_pos, (int) cfg.nd, MPI_DOUBLE,
                             MAIN_RANK, MSG_POS, MPI_COMM_WORLD);
                }
            }

            if (rank == MAIN_RANK && best_rank != MAIN_RANK) {
                MPI_Recv(global_best_pos, (int) cfg.nd, MPI_DOUBLE,
                         best_rank, MSG_POS, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        }

        IF_MAIN_PROC {
            print_result(global_best_min, global_best_pos, cfg.nd);
        }

        ret = EXIT_SUCCESS;
    }

    // Cleanup
    free(dim_starts);
    free(dim_sizes);
    free(global_best_pos);
    free(local_best_pos);
    free(scratch);
    sso_sharks_free(local_sharks, local_cfg.np);
    free(domain);

    MPI_Comm_free(&dim_comm);
    MPI_Finalize();

    return ret;
}

static error_t parser(int key, char *arg, struct argp_state *state) {
    struct Args *args = state->input;

    switch (key) {
        case 'D': {
            char *end;
            unsigned long long value = strtoull(arg, &end, 0);
            if (*arg == '\0' || *end != '\0') {
                argp_error(state, "Invalid dim-procs value: %s", arg);
                return EINVAL;
            }
            args->dim_procs = (size_t) value;
            return 0;
        }
        case ARGP_KEY_END:
            return 0;
    }

    return ARGP_ERR_UNKNOWN;
}
