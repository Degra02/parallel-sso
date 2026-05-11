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

#include "sso/parse_args.h"
#include "sso/ofuncs.h"
#include "sso/sso.h"
#include "sso/utils.h"
#include "common/utils.h"

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <mpi.h>


/**
 * @brief Serial algorithm v2 entrypoint.
 */
int main(int argc, char *argv[]) {
    // TODO: argp allows to combine different parsers. We should be able to
    // define serial/parallel specific arguments without rewriting everything.
    struct SSOConfig cfg;
    if (parse_args(argc, argv, &cfg) != 0) {
        return EXIT_FAILURE;
    }

    if (0 != MPI_Init(&argc, &argv)) return EXIT_FAILURE;

    int size, rank;
    if (0 != MPI_Comm_size(MPI_COMM_WORLD, &size)) return EXIT_FAILURE;
    if (0 != MPI_Comm_rank(MPI_COMM_WORLD, &rank)) return EXIT_FAILURE;

    // Seed the PRNG to have reproducible runs. 0 for time-based randomness.
    srand((cfg.seed == 0 ? (unsigned) time(NULL) : (unsigned) cfg.seed) + (unsigned)rank);

    IF_MAIN_PROC
        print_info(&cfg, "MPI Sharks");

    // Compute local population size (block distribution)
    size_t base = cfg.np / (size_t)size;
    size_t rem = cfg.np % (size_t)size;
    size_t local_np = base + (rank < (int)rem ? 1 : 0);

    // Domain bounds.
    struct Interval *domain = obj_alloc_domain_bounds(cfg.obj, cfg.nd);

    // Per-rank config copy with adjusted population
    struct SSOConfig local_cfg = cfg;
    local_cfg.np = local_np;

    // Allocate local sharks
    struct Shark *local_sharks = sso_sharks_alloc(domain, &local_cfg);
    double *scratch = calloc(cfg.nd, sizeof(double));
    double *local_best_pos = calloc(cfg.nd, sizeof(double));
    double *global_best_pos = calloc(cfg.nd, sizeof(double));

    int ret;
    if (domain == NULL || local_sharks == NULL || scratch == NULL || local_best_pos == NULL || global_best_pos == NULL) {
        perror("Malloc error");
        ret = EXIT_FAILURE;
    } else {
        // Local best for this rank. Minimisation problem.
        double local_best_min = INFINITY;
        double global_best_min = INFINITY;

        BENCHMARK(total_start, "Total time") {

            for (size_t shark = 0; shark < local_cfg.np; ++shark) {
                struct Shark *shark_ptr = &local_sharks[shark];

                // Perform k_max movement stages.
                for (size_t k = 0; k < local_cfg.k_max; ++k) {
                    // Update shark speed.
                    double R1 = utils_rand(0, 1); // paper random value
                    double R2 = utils_rand(0, 1); // paper random value

                    for (size_t dim = 0; dim < local_cfg.nd; ++dim) {
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

                    // NOTE: Sync barrier, everyone needs to compute the derivative
                    //       before the position is updated.
                    for (size_t dim = 0; dim < local_cfg.nd; ++dim) {
                        // Update shark position with a forward movement.
                        // Assume Δt = 1.
                        shark_ptr->position[dim] = utils_clamp(
                                shark_ptr->position[dim] + shark_ptr->speed[dim],
                                &domain[dim]);
                    }

                    // Try to teleport the shark to a better alternative.
                    // Current best candidate, initialized with the current shark position.
                    double *candidate = scratch;
                    double best = OF(shark_ptr->position,
                                                        local_cfg.nd, local_cfg.obj);
                    double best_r3 = 0.0;

                    for (uint32_t m = 0; m < local_cfg.rotations; ++m) {
                        double r3 = utils_rand(-1, 1);
                        // "Rotate" around the shark position.
                        for (size_t dim = 0; dim < local_cfg.nd; ++dim) {
                            candidate[dim] = utils_clamp(
                                        shark_ptr->position[dim] * (1 + r3),
                                        &domain[dim]);
                        }

                        // Update position if better than the current one.
                        double val = OF(candidate, local_cfg.nd, local_cfg.obj);
                        if (val > best) {
                            best = val;
                            best_r3 = r3;
                        }
                    }

                    if (best_r3 != 0.0) {
                        for (size_t dim = 0; dim < local_cfg.nd; ++dim) {
                            shark_ptr->position[dim] = utils_clamp(
                                        shark_ptr->position[dim] * (1 + best_r3),
                                        &domain[dim]);
                        }
                    }

                    // If we have an all-time best value for this rank, update it.
                    double cur_min = -best;
                    if (cur_min < local_best_min) {
                        local_best_min = cur_min;
                        memcpy(local_best_pos, shark_ptr->position, local_cfg.nd * sizeof(double));
                    }
                }
            }


            // Barrier to ensure all ranks have finished their local search before reduction.
            MPI_Barrier(MPI_COMM_WORLD);

            // Use an allreduce with MINLOC so every rank learns the global minimum and its rank.
            struct { double value; int rank; } local_pair, global_pair;
            local_pair.value = local_best_min;
            local_pair.rank = rank;

            MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

            int best_rank = global_pair.rank;
            if (rank == 0) {
                global_best_min = global_pair.value;
            }

            // Only the best rank sends its best position to root, since everyone knows best_rank from Allreduce.
            if (rank == best_rank) {
                if (rank == 0) {
                    memcpy(global_best_pos, local_best_pos, cfg.nd * sizeof(double));
                } else {
                    MPI_Send(local_best_pos, (int)cfg.nd, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }

            if (rank == 0 && best_rank != 0) {
                MPI_Recv(global_best_pos, (int)cfg.nd, MPI_DOUBLE, best_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        IF_MAIN_PROC
            print_result(global_best_min, global_best_pos, cfg.nd);

        ret = EXIT_SUCCESS;
    }

    // Cleanup
    free(global_best_pos);
    free(local_best_pos);
    free(scratch);
    sso_sharks_free(local_sharks, local_cfg.np);
    free(domain);

    MPI_Finalize();

    return ret;
}
