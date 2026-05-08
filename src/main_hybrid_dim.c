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
#include <omp.h>

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
    unsigned int seed_base = (unsigned int)(cfg.seed == 0 ? (unsigned) time(NULL) : (unsigned) cfg.seed) + (unsigned)rank;
    srand(seed_base);

    IF_MAIN_PROC {
        print_info(&cfg, "Hybrid Dimensions");
    }

    // Domain bounds.
    struct Interval *domain = obj_alloc_domain_bounds(cfg.obj, cfg.nd);

    // Alloc population sharks at random positions in the domain, with 0 speed.
    struct Shark *sharks = sso_sharks_alloc(domain, &cfg);

    int omp_threads = (cfg.threads == 0) ? omp_get_max_threads() : (int) cfg.threads;

    // Per-thread scratch storage for rotational search.
    double *scratch_all = calloc((size_t)omp_threads * cfg.nd, sizeof(double));

    // The best position found up to date.
    double *best_pos = calloc(cfg.nd, sizeof(double));

    int *scatter_sizes = calloc(size, sizeof(int));
    int *scatter_starts = calloc(size, sizeof(int));

    int ret;
    if (domain == NULL || sharks == NULL || scratch_all == NULL || best_pos == NULL
        || scatter_sizes == NULL || scatter_starts == NULL) {
        perror("Malloc error");
        free(scratch_all);
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

            #pragma omp parallel num_threads(omp_threads)
            {
                int tid = omp_get_thread_num();
                unsigned int seed = (unsigned int)(cfg.seed == 0 ? (unsigned) time(NULL) : (unsigned) cfg.seed) + tid;
                double *thread_scratch = &scratch_all[(size_t)tid * cfg.nd];

                for (size_t shark = 0; shark < cfg.np; ++shark) {
                    struct Shark *shark_ptr = &sharks[shark];

                    // Perform k_max movement stages.
                    for (size_t k = 0; k < cfg.k_max; ++k) {
                        double R1 = thread_rand_r(&seed, 0.0, 1.0);
                        double R2 = thread_rand_r(&seed, 0.0, 1.0);

                        // Speed update: parallelize over local dimensions.
                        #pragma omp for schedule(static) nowait
                        for (size_t dim = start_dim; dim < end_dim; ++dim) {
                            double v_prev = shark_ptr->speed[dim];

                            double derivative = eval_derivative(shark_ptr->position,
                                                        cfg.nd, cfg.obj, dim);
                            double grad_term = cfg.eta * R1 * derivative;
                            double mom_term  = cfg.alpha * R2 * v_prev;
                            shark_ptr->speed[dim] = grad_term + mom_term;

                            if (fabs(v_prev) >= 1e-15) {
                                double limit = cfg.beta * v_prev;
                                if (fabs(shark_ptr->speed[dim]) > fabs(limit)) {
                                    shark_ptr->speed[dim] = limit;
                                }
                            }
                        }

                        // Position update: parallelize over local dimensions.
                        #pragma omp for schedule(static) nowait
                        for (size_t dim = start_dim; dim < end_dim; ++dim) {
                            shark_ptr->position[dim] = utils_clamp(
                                    shark_ptr->position[dim] + shark_ptr->speed[dim],
                                    &domain[dim]);
                        }

                        #pragma omp barrier
                        #pragma omp single
                        {
                            MPI_Allgatherv(shark_ptr->position + start_dim,
                                end_dim - start_dim, MPI_DOUBLE,
                                shark_ptr->position, scatter_sizes,
                                scatter_starts, MPI_DOUBLE,
                                MPI_COMM_WORLD);
                        }
                        #pragma omp barrier

                        // Try to teleport the shark to a better alternative.
                        // Single thread evaluates and tracks best rotation
                        #pragma omp single
                        {
                            double best = OF(shark_ptr->position, cfg.nd, cfg.obj);
                            double best_r3 = 0.0;

                            for (uint32_t m = 0; m < cfg.rotations; ++m) {
                                double r3 = thread_rand_r(&seed, -1.0, 1.0);

                                for (size_t dim = 0; dim < cfg.nd; ++dim) {
                                    thread_scratch[dim] = utils_clamp(
                                        shark_ptr->position[dim] * (1 + r3),
                                        &domain[dim]);
                                }

                                double val = OF(thread_scratch, cfg.nd, cfg.obj);
                                if (val > best) {
                                    best = val;
                                    best_r3 = r3;
                                }
                            }

                            if (best_r3 != 0.0) {
                                // Update position with best rotation
                                for (size_t dim = 0; dim < cfg.nd; ++dim) {
                                    shark_ptr->position[dim] = utils_clamp(
                                            shark_ptr->position[dim] * (1 + best_r3),
                                            &domain[dim]);
                                }
                            }

                            // If we have an all-time best value, update the current one.
                            double cur_min = -best;
                            if (cur_min < best_min) {
                                best_min = cur_min;
                                memcpy(best_pos, shark_ptr->position, cfg.nd * sizeof(double));
                            }
                        }
                    } // end stages
                } // end sharks
            } // end parallel region

            // Ensure everyone has finished.
            MPI_Barrier(MPI_COMM_WORLD);

        } // end benchmark

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
    free(scratch_all);
    sso_sharks_free(sharks, cfg.np);
    free(domain);
    return ret;
}
