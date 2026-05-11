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

#include <argp.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

enum Msgs {
    MSG_POS = 1
};

struct Args {
    size_t thread_num; /**< Number of threads, 0 to use all available threads. */
};

static const struct argp argp;

/**
 * @brief Serial algorithm v2 entrypoint.
 */
int main(int argc, char *argv[]) {
    struct SSOConfig cfg;
    struct Args args;
    if (parse_args_extend(argc, argv, &cfg, &argp, &args) != 0) {
        return EXIT_FAILURE;
    }

    if (0 != MPI_Init(&argc, &argv)) return EXIT_FAILURE;

    int size, rank;
    if (0 != MPI_Comm_size(MPI_COMM_WORLD, &size)) return EXIT_FAILURE;
    if (0 != MPI_Comm_rank(MPI_COMM_WORLD, &rank)) return EXIT_FAILURE;

    // Seed the PRNG to have reproducible runs. 0 for time-based randomness.
    unsigned seed_base = cfg.seed == 0
                         ? (unsigned) time(NULL) + (rank << 16)
                         : (unsigned) cfg.seed + rank;

    IF_MAIN_PROC {
        print_info(&cfg, "Hybrid Sharks");
    }

    size_t max_threads = omp_get_max_threads();
    size_t thread_num = args.thread_num != 0 ? args.thread_num : max_threads;
    printf("Thread num: %lu\n", thread_num);

    if (thread_num > max_threads) {
        fprintf(stderr, "Cannot use %lu threads, max is %lu",
                thread_num, max_threads);
        return EXIT_FAILURE;
    }


    // Domain bounds.
    struct Interval *domain = obj_alloc_domain_bounds(cfg.obj, cfg.nd);

    // Alloc population sharks at random positions in the domain, with 0 speed.
    struct Shark *sharks = sso_sharks_alloc(domain, &cfg);

    // Scratch array to avoid allocations in loops.
    double **scratch_pool = calloc_matrix(thread_num, cfg.nd,
                                          sizeof(double));

    // The best position found up to date.
    double *best_pos = calloc(cfg.nd, sizeof(double));
    double **best_pos_pool = calloc_matrix(thread_num, cfg.nd,
                                           sizeof(double) * thread_num);

    int ret;
    if (domain == NULL || sharks == NULL || scratch_pool == NULL
        || best_pos == NULL || best_pos_pool == NULL) {
        perror("Malloc error");
        ret = EXIT_FAILURE;
    } else {
        // Global best. Minimisation problem.
        double best_min = INFINITY;

        BENCHMARK(total_start, "Total time") {

            #pragma omp parallel num_threads(thread_num)
            {
                size_t tid = omp_get_thread_num();
                unsigned int seed = seed_base + tid;
                double local_best_min = INFINITY;
                double *local_best_pos = best_pos_pool[tid];
                double *scratch = scratch_pool[tid];

                #pragma omp for schedule(static)
                for (size_t shark = rank; shark < cfg.np; shark += size) {
                    struct Shark *shark_ptr = &sharks[shark];

                    // Perform k_max movement stages.
                    for (size_t k = 0; k < cfg.k_max; ++k) {
                        // Update shark speed.
                        double R1 = thread_rand_r(&seed, 0, 1); // paper random value
                        double R2 = thread_rand_r(&seed, 0, 1); // paper random value

                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
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

                        // NOTE: Sync barrier, everyone needs to compute the derivative
                        //       before the position is updated.
                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
                            // Update shark position with a forward movement.
                            // Assume Δt = 1.
                            shark_ptr->position[dim] = utils_clamp(
                                    shark_ptr->position[dim] + shark_ptr->speed[dim],
                                    &domain[dim]);
                        }

                        // Try to teleport the shark to a better alternative.
                        // Current best candidate, initialized with the current shark position.
                        double *candidate = scratch;
                        double best = OF(shark_ptr->position, cfg.nd, cfg.obj);
                        double best_r3 = 0.0;

                        for (uint32_t m = 0; m < cfg.rotations; ++m) {
                            double r3 = thread_rand_r(&seed, -1, 1);
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
                            for (size_t dim = 0; dim < cfg.nd; ++dim) {
                                shark_ptr->position[dim] = utils_clamp(
                                            shark_ptr->position[dim] * (1 + best_r3),
                                            &domain[dim]);
                            }
                        }

                        // If we have an all-time best value, update the current one.
                        double cur_min = -best;
                        if (cur_min < local_best_min) {
                            local_best_min = cur_min;
                            memcpy(local_best_pos, shark_ptr->position,
                                    cfg.nd * sizeof(double));
                        }
                    }

                }

                #pragma omp critical
                if (local_best_min < best_min) {
                    best_min = local_best_min;
                    memcpy(best_pos, local_best_pos, cfg.nd * sizeof(double));
                }

            }

            double global_best = INFINITY;
            MPI_Allreduce(&best_min, &global_best, 1,
                          MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);


            IF_MAIN_PROC {
                if (best_min != global_best) {
                    MPI_Recv(best_pos, cfg.nd, MPI_DOUBLE,
                             MPI_ANY_SOURCE, MSG_POS,
                             MPI_COMM_WORLD,NULL);
                }
            } else if (best_min == global_best) {
                MPI_Send(best_pos, cfg.nd, MPI_DOUBLE,
                         MAIN_RANK, MSG_POS, MPI_COMM_WORLD);
            }

            // Barrier for benchmarking purposes.
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
    free(best_pos);
    free_matrix(thread_num, scratch_pool);
    free_matrix(thread_num, best_pos_pool);
    sso_sharks_free(sharks, cfg.np);
    free(domain);
    return ret;
}

static error_t parser(int key, char *arg, struct argp_state *state);

static const struct argp_option options[] = {
    {"threads",     't', "int"  , 0, "The population size.",                    1},
    { 0 } // This is needed to "terminate" the array.
};

static const struct argp argp = {options, parser, "", NULL, 0, 0, 0};

#define RET_PARSE_BOUNDED(size, args, field, val, min, max, ...) do {          \
        char *end;                                                             \
        (args)->field = strto##size((val), &end __VA_OPT__(,) __VA_ARGS__);    \
        if (*(val) == 0 || *end != 0) {                                        \
            perror("Couldn't parse " #field);                                  \
            return -1;                                                         \
        }                                                                      \
        if (*(val) < min || *(val) > max) {                                    \
            return -1;                                                         \
        }                                                                      \
        return 0;                                                              \
    } while(0)

#define RET_PARSE(size, args, field, val, ...) do {                            \
        char *end;                                                             \
        (args)->field = strto##size((val), &end __VA_OPT__(,) __VA_ARGS__);    \
        if (*(val) == 0 || *end != 0) {                                        \
            perror("Couldn't parse " #field);                                  \
            return -1;                                                         \
        }                                                                      \
        return 0;                                                              \
    } while(0)

#define RET_PARSE_ULL(args, field, val) RET_PARSE(ull, args, field, val, 0)
#define RET_PARSE_D(args, field, val) RET_PARSE(d, args, field, val)
#define RET_PARSE_D_BOUNDED(args, field, val, min, max)\
            RET_PARSE_BOUNDED(d, args, field, val, min, max)

static error_t parser(int key, char *arg, struct argp_state *state) {
    struct SSOConfig *cfg = state->input;
    struct Args *args = cfg->extension;

    switch (key) {
        case 't':
            RET_PARSE_ULL(args, thread_num, arg);
        case ARGP_KEY_END:
            return 0;
    }

    return ARGP_ERR_UNKNOWN;
}
