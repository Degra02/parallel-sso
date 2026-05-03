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
#include <mpi.h>
#include <time.h>
#include <stdlib.h>


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
    // TODO: sync time-based seed.
    srand(cfg.seed == 0   ? (unsigned) time(NULL)
                                : (unsigned) cfg.seed);

    IF_MAIN_PROC {
        print_info(&cfg);
    }

    // Domain bounds.
    struct Interval *domain = obj_alloc_domain_bounds(cfg.obj, cfg.nd);

    // Alloc population sharks at random positions in the domain, with 0 speed.
    struct Shark *sharks = sso_sharks_alloc(domain, &cfg);

    // Scratch array to avoid allocations in loops.
    double *scratch = calloc(cfg.nd, sizeof(double));

    // The best position found up to date.
    double *best_pos = calloc(cfg.nd, sizeof(double));

    int *scatter_sizes = calloc(size, sizeof(int));
    int *scatter_starts = calloc(size, sizeof(int));

    int ret;
    if (domain == NULL || sharks == NULL || scratch == NULL || best_pos == NULL
        || scatter_sizes == NULL || scatter_starts == NULL) {
        perror("Malloc error");
        ret = EXIT_FAILURE;
    } else {
        // Global best. Minimisation problem.
        double best_min = INFINITY;
        size_t start_dim = 0, end_dim = 0;
        for (size_t i = 0; i < size; ++i) {
            size_t remainder = cfg.nd % size;
            scatter_starts[i] = cfg.nd / size * i + (i < remainder ? i : remainder);
            scatter_sizes[i] = cfg.nd / size + (i < remainder ? 1 : 0);
            if (i == rank) {
                start_dim = scatter_starts[i];
                end_dim = start_dim + scatter_sizes[i];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        BENCHMARK(total_start, "Total time") {

            for (size_t shark = 0; shark < cfg.np; ++shark) {
                struct Shark *shark_ptr = &sharks[shark];

                // Perform k_max movement stages.
                for (size_t k = 0; k < cfg.k_max; ++k) {
                    // Update shark speed.
                    double R1 = utils_rand(0, 1); // paper random value
                    double R2 = utils_rand(0, 1); // paper random value

                    for (size_t dim = start_dim; dim < end_dim; ++dim) {
                        double v_prev = shark_ptr->speed[dim];

                        double derivative = eval_derivative(shark_ptr->position,
                                                    cfg.nd, cfg.obj, dim);
                        // gradient based speed: γ·R1·gradient
                        double grad_term = cfg.eta * R1 * derivative;
                        // momentum based speed: α·R2·v_{k-1}
                        double mom_term  = cfg.alpha * R2 * v_prev;
                        shark_ptr->speed[dim] = grad_term + mom_term;

                        if (fabs(v_prev) >= 1e-15) {
                            // Limit the velocity up to β·v.
                            double limit = cfg.beta * v_prev;
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
                        end_dim - start_dim, MPI_UINT64_T,
                        shark_ptr->position, scatter_sizes,
                        scatter_starts, MPI_UINT64_T,
                        MPI_COMM_WORLD);

                    // NOTE: let everyone compute this to avoid resync.
                    // TODO: evaluate the impact of approximations.

                    // Try to teleport the shark to a better alternative.
                    // Current best candidate, initialized with the current shark position.
                    double *candidate = scratch;
                    double best = OF(shark_ptr->position,
                                     cfg.nd, cfg.obj);
                    double best_r3 = 0.0;

#if PAR_ROTATIONS
    #error "Not implemented yet"
#else
                    for (uint32_t m = 0; m < cfg.rotations; ++m) {
#endif
                        double r3 = utils_rand(-1, 1);
                        // "Rotate" around the shark position.
#if PAR_ALL_DIM
                        for (size_t dim = start_dim; dim < end_dim; ++dim) {
#else
                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
#endif
                            candidate[dim] = utils_clamp(
                                        shark_ptr->position[dim] * (1 + r3),
                                        &domain[dim]);
                        }
#if PAR_ALL_DIM
                        // TODO: gather or allgather.
                        MPI_Allgatherv(candidate + start_dim,
                            end_dim - start_dim, MPI_UINT64_T,
                            candidate, scatter_sizes,
                            scatter_starts, MPI_UINT64_T,
                            MPI_COMM_WORLD);
#endif
                        // "Rotate" around the shark position.
                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
                            candidate[dim] = utils_clamp(
                                        shark_ptr->position[dim] * (1 + r3),
                                        &domain[dim]);
                        }

                        // Update position if better than the current one.
                        double val = OF(candidate, cfg.nd, cfg.obj);
                        if (val > best) {
                            best = val;
                            best_r3 = r3;
                        }
                    }

                    if (best_r3 != 0.0) {
#if PAR_ALL_DIM
                        for (size_t dim = start_dim; dim < end_dim; ++dim) {
#else
                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
#endif
                            shark_ptr->position[dim] = utils_clamp(
                                        shark_ptr->position[dim] * (1 + best_r3),
                                        &domain[dim]);
                        }
#if PAR_ALL_DIM
                        // TODO: gather or allgather.
                        MPI_Allgatherv(shark_ptr->position + start_dim,
                            end_dim - start_dim, MPI_UINT64_T,
                            shark_ptr->position, scatter_sizes,
                            scatter_starts, MPI_UINT64_T,
                            MPI_COMM_WORLD);
#endif
                    }

                    // If we have an all-time best value, update the current one.
                    double cur_min = -best;
                    if (cur_min < best_min) {
                        best_min = cur_min;
                        memcpy(best_pos, shark_ptr->position, cfg.nd * sizeof(double));
                    }

                }
            }

            // Ensure everyone has finished.
            MPI_Barrier(MPI_COMM_WORLD);

        }

        // Print final result.
        IF_MAIN_PROC {
            print_result(best_min, best_pos, cfg.nd);
        }

        ret = EXIT_SUCCESS;
    }

    MPI_Finalize();

    // Cleanup
    free(scatter_starts);
    free(scatter_sizes);
    free(best_pos);
    free(scratch);
    sso_sharks_free(sharks, cfg.np);
    free(domain);
    return ret;
}
