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
#include <string.h>

enum Msgs {
    MSG_POS = 1
};

struct Args {
    size_t thread_num; /**< Number of threads, 0 to use all available threads */
};

static const struct argp argp;

/**
 * @brief Hybrid algorithm with MPI over sharks and OpenMP over dimensions
 */
int main(int argc, char *argv[]) {
    struct SSOConfig cfg;
    struct Args args = {0};
    if (parse_args_extend(argc, argv, &cfg, &argp, &args) != 0) {
        return EXIT_FAILURE;
    }

    if (0 != MPI_Init(&argc, &argv)) return EXIT_FAILURE;

    int size, rank;
    if (0 != MPI_Comm_size(MPI_COMM_WORLD, &size)) return EXIT_FAILURE;
    if (0 != MPI_Comm_rank(MPI_COMM_WORLD, &rank)) return EXIT_FAILURE;

    // Seed the PRNG to have reproducible runs. 0 for time-based randomness
    unsigned init_seed = cfg.seed == 0 ? (unsigned) time(NULL)
                                       : (unsigned) cfg.seed;
    unsigned seed = init_seed + (unsigned) rank;
    srand(init_seed);

    IF_MAIN_PROC {
        print_info(&cfg, "Hybrid Hybrid");
    }

    size_t max_threads = omp_get_max_threads();
    size_t thread_num = args.thread_num != 0 ? args.thread_num : max_threads;

    if (thread_num > max_threads) {
        fprintf(stderr, "Cannot use %lu threads, max is %lu\n",
                thread_num, max_threads);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Domain bounds
    struct Interval *domain = obj_alloc_domain_bounds(cfg.obj, cfg.nd);

    // Alloc population sharks at random positions in the domain, with 0 speed
    struct Shark *sharks = sso_sharks_alloc(domain, &cfg);

    // Scratch array to avoid allocations in loops
    double *scratch = calloc(cfg.nd, sizeof(double));

    // The best position found up to date.
    double *best_pos = calloc(cfg.nd, sizeof(double));

    int ret;
    if (domain == NULL || sharks == NULL || scratch == NULL || best_pos == NULL) {
        perror("Malloc error");
        ret = EXIT_FAILURE;
    } else {
        // Local best. Minimisation problem
        double best_min = INFINITY;

        double R1 = 0.0;
        double R2 = 0.0;
        double r3 = 0.0;
        double best = 0.0;
        double best_r3 = 0.0;

        BENCHMARK(total_start, "Total time") {

            #pragma omp parallel num_threads((int) thread_num) default(none) \
                shared(cfg, domain, sharks, scratch, best_pos, best_min, \
                       R1, R2, r3, best, best_r3, seed, rank, size)
            {
                for (size_t shark = (size_t) rank; shark < cfg.np;
                     shark += (size_t) size) {
                    struct Shark *shark_ptr = &sharks[shark];

                    // Perform k_max movement stages
                    for (size_t k = 0; k < cfg.k_max; ++k) {

                        #pragma omp single
                        {
                            R1 = thread_rand_r(&seed, 0, 1);
                            R2 = thread_rand_r(&seed, 0, 1);
                        }

                        // Update shark speed
                        #pragma omp for schedule(static)
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
                                // Limit the velocity up to β·v
                                double limit = cfg.beta * v_prev;
                                if (fabs(shark_ptr->speed[dim]) > fabs(limit)) {
                                    shark_ptr->speed[dim] = limit;
                                }
                            }
                        }

                        // Forward movement
                        #pragma omp for schedule(static)
                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
                            shark_ptr->position[dim] = utils_clamp(
                                    shark_ptr->position[dim] + shark_ptr->speed[dim],
                                    &domain[dim]);
                        }

                        // Current best candidate
                        #pragma omp single
                        {
                            best = OF(shark_ptr->position, cfg.nd, cfg.obj);
                            best_r3 = 0.0;
                        }

                        for (uint32_t m = 0; m < cfg.rotations; ++m) {
                            #pragma omp single
                            {
                                r3 = thread_rand_r(&seed, -1, 1);
                            }

                            // Rotate around the shark position
                            #pragma omp for schedule(static)
                            for (size_t dim = 0; dim < cfg.nd; ++dim) {
                                scratch[dim] = utils_clamp(
                                            shark_ptr->position[dim] * (1 + r3),
                                            &domain[dim]);
                            }

                            // Update position if better than the current one
                            #pragma omp single
                            {
                                double val = OF(scratch, cfg.nd, cfg.obj);
                                if (val > best) {
                                    best = val;
                                    best_r3 = r3;
                                }
                            }
                        }

                        if (best_r3 != 0.0) {
                            #pragma omp for schedule(static)
                            for (size_t dim = 0; dim < cfg.nd; ++dim) {
                                shark_ptr->position[dim] = utils_clamp(
                                            shark_ptr->position[dim] * (1 + best_r3),
                                            &domain[dim]);
                            }
                        }

                        // If we have an all-time best value, update the current one
                        #pragma omp single
                        {
                            double cur_min = -best;
                            if (cur_min < best_min) {
                                best_min = cur_min;
                                memcpy(best_pos, shark_ptr->position,
                                       cfg.nd * sizeof(double));
                            }
                        }
                    }
                }
            }

            struct {
                double val;
                int rank;
            } local_best, global_best;

            local_best.val = best_min;
            local_best.rank = rank;

            MPI_Allreduce(&local_best, &global_best, 1,
                          MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

            if (rank == MAIN_RANK) {
                if (global_best.rank != MAIN_RANK) {
                    MPI_Recv(best_pos, cfg.nd, MPI_DOUBLE,
                             global_best.rank, MSG_POS,
                             MPI_COMM_WORLD, NULL);
                }
                best_min = global_best.val;
            } else if (rank == global_best.rank) {
                MPI_Send(best_pos, cfg.nd, MPI_DOUBLE,
                         MAIN_RANK, MSG_POS, MPI_COMM_WORLD);
            }

            // Barrier for benchmarking purposes
            MPI_Barrier(MPI_COMM_WORLD);
        }

        IF_MAIN_PROC {
            print_result(best_min, best_pos, cfg.nd);
        }

        ret = EXIT_SUCCESS;
    }

    MPI_Finalize();

    // Cleanup
    free(best_pos);
    free(scratch);
    sso_sharks_free(sharks, cfg.np);
    free(domain);
    return ret;
}

static error_t parser(int key, char *arg, struct argp_state *state);

static const struct argp_option options[] = {
    {"threads",     't', "int"  , 0, "The population size.",                    1},
    { 0 } // This is needed to "terminate" the array
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
    struct Args *args = state->input;

    switch (key) {
        case 't':
            RET_PARSE_ULL(args, thread_num, arg);
        case ARGP_KEY_END:
            return 0;
    }

    return ARGP_ERR_UNKNOWN;
}