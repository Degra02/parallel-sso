#include "sso/parse_args.h"
#include "sso/ofuncs.h"
#include "sso/sso.h"
#include "sso/utils.h"

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stdio.h>
#include "common/utils.h"

#ifndef MAX_THREADS
    #define MAX_THREADS 8
#endif

static inline double thread_rand_r(unsigned int *seedp, double min, double max) {
    return (double) rand_r(seedp) / ((double) RAND_MAX + 1.0) * (max - min) + min;
}


/**
 * @brief OpenMP parallel sharks algorithm entrypoint.
 */
int main(int argc, char *argv[]) {
    struct SSOConfig cfg;
    if (parse_args(argc, argv, &cfg) != 0) {
        return EXIT_FAILURE;
    }

    print_info(&cfg);

    // Domain bounds.
    struct Interval *domain = obj_alloc_domain_bounds(cfg.obj, cfg.nd);

    // Seed global PRNG for initial population
    unsigned int seed_base = (unsigned int)(cfg.seed == 0 ? (unsigned) time(NULL) : (unsigned) cfg.seed);
    srand(seed_base);

    // Allocate population sharks at random positions in the domain, with 0 speed.
    struct Shark *sharks = sso_sharks_alloc(domain, &cfg);

    // Global best result
    double *best_pos = calloc(cfg.nd, sizeof(double));
    double best_min = INFINITY;

    if (domain == NULL || sharks == NULL || best_pos == NULL) {
        perror("Malloc error");
        free(best_pos);
        sso_sharks_free(sharks, cfg.np);
        free(domain);
        return EXIT_FAILURE;
    }

    // Prepare per-thread scratch storage and per-thread best storage.
    int max_threads = omp_get_max_threads();
    double *scratch_all = calloc((size_t)max_threads * cfg.nd, sizeof(double));
    double *thread_best_pos_all = calloc((size_t)max_threads * cfg.nd, sizeof(double));

    if (scratch_all == NULL || thread_best_pos_all == NULL) {
        perror("Malloc error");
        free(scratch_all);
        free(thread_best_pos_all);
        free(best_pos);
        sso_sharks_free(sharks, cfg.np);
        free(domain);
        return EXIT_FAILURE;
    }

    // Parallelize over sharks; each thread evolves its sharks through all stages.
    BENCHMARK_OPENMP(total_start, "Total OpenMP time") {
        #pragma omp parallel num_threads(MAX_THREADS)
        {
            int tid = omp_get_thread_num();
            unsigned int seed = seed_base + (unsigned)tid;
            double *local_scratch = &scratch_all[(size_t)tid * cfg.nd];
            double *local_best_pos = &thread_best_pos_all[(size_t)tid * cfg.nd];
            double local_best_min = INFINITY;

            #pragma omp for schedule(static)
            for (size_t shark = 0; shark < cfg.np; ++shark) {
                struct Shark *shark_ptr = &sharks[shark];

                // Perform k_max movement stages for this shark.
                for (size_t k = 0; k < cfg.k_max; ++k) {
                    // Update shark speed using thread-local RNG
                    double R1 = thread_rand_r(&seed, 0.0, 1.0);
                    double R2 = thread_rand_r(&seed, 0.0, 1.0);

                    for (size_t dim = 0; dim < cfg.nd; ++dim) {
                        double v_prev    = shark_ptr->speed[dim];

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

                    // Update position
                    for (size_t dim = 0; dim < cfg.nd; ++dim) {
                        shark_ptr->position[dim] = utils_clamp(
                                shark_ptr->position[dim] + shark_ptr->speed[dim],
                                &domain[dim]);
                    }

                    // Rotational local search using thread-local scratch and RNG
                    double *candidate = local_scratch;
                    double best = OF(shark_ptr->position, cfg.nd, cfg.obj);
                    double best_r3 = 0.0;

                    for (uint32_t m = 0; m < cfg.rotations; ++m) {
                        double r3 = thread_rand_r(&seed, -1.0, 1.0);
                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
                            candidate[dim] = utils_clamp(
                                    shark_ptr->position[dim] * (1 + r3),
                                    &domain[dim]);
                        }

                        double val = OF(candidate, cfg.nd, cfg.obj);
                        if (val > best) {
                            best = val;
                            best_r3 = r3;
                        }
                    }

                    if (best_r3 != 0.0) {
                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
                            shark_ptr->position[dim] = utils_clamp(
                                        shark_ptr->position[dim] * (1 + best_r3),
                                        &domain[dim]);
                        }
                    }

                    // If we have an all-time best value, update thread-local best
                    double cur_min = -best;
                    if (cur_min < local_best_min) {
                        local_best_min = cur_min;
                        memcpy(local_best_pos, shark_ptr->position, cfg.nd * sizeof(double));
                    }
                } // end stages
            } // end for sharks

            // Reduce per-thread best into shared global best (single sync point at end).
            #pragma omp critical
            {
                if (local_best_min < best_min) {
                    best_min = local_best_min;
                    memcpy(best_pos, local_best_pos, cfg.nd * sizeof(double));
                }
            }
        } // end parallel region
    } // end benchmark

    print_result(best_min, best_pos, cfg.nd);

    // Cleanup
    free(scratch_all);
    free(thread_best_pos_all);
    free(best_pos);
    sso_sharks_free(sharks, cfg.np);
    free(domain);
    return EXIT_SUCCESS;
}
