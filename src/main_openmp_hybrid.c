#include "sso/parse_args.h"
#include "sso/ofuncs.h"
#include "sso/sso.h"
#include "sso/utils.h"
#include "common/utils.h"

#include <argp.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

struct Args {
    size_t threads;
    size_t shark_threads;
};

static const struct argp argp;

/**
 * @brief OpenMP parallel sharks algorithm entrypoint.
 */
int main(int argc, char *argv[]) {
    int ret = EXIT_FAILURE;
    struct SSOConfig cfg;
    struct Args args = {0};

    if (parse_args_extend(argc, argv, &cfg, &argp, &args) != 0) {
        return EXIT_FAILURE;
    }

    print_info(&cfg, "OpenMP Sharks");

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
    size_t omp_threads = (args.threads == 0) ? omp_get_max_threads() : (int) args.threads;
    if (omp_get_max_threads() < omp_threads) {
        fprintf(stderr, "Too many threads %lu/%d\n", omp_threads, omp_get_max_threads());
        return EXIT_FAILURE;
    }

    size_t shark_threads_num = args.shark_threads;
    if (omp_threads % shark_threads_num != 0) {
        fprintf(stderr, "The number of threads for sharks and dims must multiply to %lu\n", omp_threads);
        return EXIT_FAILURE;
    }
    size_t dim_threads_num = omp_threads / args.shark_threads;
    omp_set_dynamic(0);
    omp_set_max_active_levels(2);

    double *scratch_all = calloc((size_t)omp_threads * cfg.nd, sizeof(double));
    double *thread_best_pos_all = calloc((size_t)omp_threads * cfg.nd, sizeof(double));

    if (scratch_all == NULL || thread_best_pos_all == NULL) {
        perror("Malloc error");
        ret = EXIT_FAILURE;
        goto cleanup;
    }

    // Nested OpenMP: outer team splits sharks, inner teams split dimensions.
    BENCHMARK_OPENMP(total_start, "Total OpenMP time") {
        #pragma omp parallel num_threads(shark_threads_num) default(none) \
            shared(cfg, domain, sharks, scratch_all, thread_best_pos_all, \
                   best_min, best_pos, seed_base, dim_threads_num)
        {
            int outer_tid = omp_get_thread_num();
            unsigned int seed = seed_base + ((unsigned int)outer_tid << 16);
            double *local_scratch = &scratch_all[(size_t)outer_tid * cfg.nd];
            double *local_best_pos = &thread_best_pos_all[(size_t)outer_tid * cfg.nd];
            double local_best_min = INFINITY;

            #pragma omp for schedule(static)
            for (size_t shark = 0; shark < cfg.np; ++shark) {
                struct Shark *shark_ptr = &sharks[shark];

                // Perform k_max movement stages for this shark.
                for (size_t k = 0; k < cfg.k_max; ++k) {
                    // Update shark speed using thread-local RNG
                    double R1 = thread_rand_r(&seed, 0.0, 1.0);
                    double R2 = thread_rand_r(&seed, 0.0, 1.0);

                    #pragma omp parallel for num_threads(dim_threads_num) \
                        default(none) shared(cfg, domain, shark_ptr, dim_threads_num) \
                        firstprivate(R1, R2) schedule(static)
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

                    // NOTE: the implicit barrier is important since modifications
                    // to position must happen after eval_derivative calls.

                    // Update position
                    #pragma omp parallel for num_threads(dim_threads_num) \
                        default(none) shared(cfg, domain, shark_ptr, dim_threads_num) \
                        schedule(static)
                    for (size_t dim = 0; dim < cfg.nd; ++dim) {
                        shark_ptr->position[dim] = utils_clamp(
                                shark_ptr->position[dim] + shark_ptr->speed[dim],
                                &domain[dim]);
                    }

                    // NOTE: the implicit barrier is important since calls to OF
                    // must happen after position is completely updated.

                    // Rotational local search using thread-local scratch and RNG
                    double *candidate = local_scratch;
                    double best = OF(shark_ptr->position, cfg.nd, cfg.obj);
                    double best_r3 = 0.0;

                    for (uint32_t m = 0; m < cfg.rotations; ++m) {
                        double r3 = thread_rand_r(&seed, -1.0, 1.0);

                        #pragma omp parallel for num_threads(dim_threads_num) \
                            default(none) shared(cfg, domain, shark_ptr, candidate, dim_threads_num) \
                            firstprivate(r3) schedule(static)
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
                        #pragma omp parallel for num_threads(dim_threads_num) \
                            default(none) shared(cfg, domain, shark_ptr, dim_threads_num) \
                            firstprivate(best_r3) schedule(static)
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
        ret = EXIT_SUCCESS;
    } // end benchmark

    print_result(best_min, best_pos, cfg.nd);

cleanup:
    // Cleanup
    free(scratch_all);
    free(thread_best_pos_all);
    free(best_pos);
    sso_sharks_free(sharks, cfg.np);
    free(domain);
    return ret;
}


static error_t parser(int key, char *arg, struct argp_state *state);

static const struct argp_option options[] = {
    {"threads",     't', "int"  , 0, "The number of threads to use",                                        1},
    {"tsharks",     'x', "int"  , 0, "The number of threads to allocate for the sharks parallelization",    1},
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
    struct Args *args = state->input;

    switch (key) {
        case 't':
            RET_PARSE_ULL(args, threads, arg);
        case 'x':
            RET_PARSE_ULL(args, shark_threads, arg);
        case ARGP_KEY_END:
            if (args->shark_threads == 0) {
                argp_failure(state, 1, 0, "Options -x is required. See --help for more information.");
                exit(EXIT_FAILURE);
            }
            return 0;
    }

    return ARGP_ERR_UNKNOWN;
}
