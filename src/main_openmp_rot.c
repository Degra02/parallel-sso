#include "sso/parse_args.h"
#include "sso/ofuncs.h"
#include "sso/sso.h"
#include "sso/utils.h"
#include "common/utils.h"

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

struct Args {
    size_t threads;
};

struct RotationBest{
    double best;
    double best_r3;
};

// OpenMP reduction for finding the best rotation across threads. We use a struct to keep both the best value and the corresponding r3.
#pragma omp declare reduction(rot_best : struct RotationBest : \
    omp_out = (omp_in.best > omp_out.best ? omp_in : omp_out)) \
    initializer(omp_priv = (struct RotationBest){ -INFINITY, 0.0 })

static const struct argp argp;

/**
 * @brief OpenMP parallel sharks algorithm entrypoint.
 */
int main(int argc, char *argv[]) {
    struct SSOConfig cfg;
    struct Args args = {0};

    if (parse_args_extend(argc, argv, &cfg, &argp, &args) != 0) {
        return EXIT_FAILURE;
    }

    print_info(&cfg, "OpenMP Sharks");
    printf("threads=%lu ", args.threads);

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

    // Prepare per-thread scratch storage for the parallel rotation search.
    size_t omp_threads = (args.threads == 0) ? omp_get_max_threads() : (size_t) args.threads;
    if (omp_get_max_threads() < omp_threads) {
        fprintf(stderr, "Too many threads %lu/%lu\n", omp_threads, (size_t)omp_get_max_threads());
        return EXIT_FAILURE;
    }

    double *scratch_all = calloc(omp_threads * cfg.nd, sizeof(double));

    if (scratch_all == NULL) {
        perror("Malloc error");
        free(scratch_all);
        free(best_pos);
        sso_sharks_free(sharks, cfg.np);
        free(domain);
        return EXIT_FAILURE;
    }

    // Shared per-stage state, written serially and read by all threads in the rotation loop.
    double best = 0.0, best_r3 = 0.0;
    struct RotationBest rot_best = { .best = -INFINITY, .best_r3 = 0.0 };

    BENCHMARK_OPENMP(total_start, "Total OpenMP time") {

        #pragma omp parallel num_threads(omp_threads) default(none) \
            shared(cfg, domain, sharks, scratch_all, best_pos, best_min, \
                   seed_base, best, best_r3, rot_best)
        {
            int tid = omp_get_thread_num();
            unsigned int seed = seed_base + ((unsigned int)tid << 16);
            double *candidate = &scratch_all[(size_t)tid * cfg.nd];

            for (size_t shark = 0; shark < cfg.np; ++shark) {
                struct Shark *shark_ptr = &sharks[shark];

                for (size_t k = 0; k < cfg.k_max; ++k) {

                    // Speed update, position update, and rotation state init are serial.
                    #pragma omp single
                    {
                        double R1 = utils_rand(0.0, 1.0);
                        double R2 = utils_rand(0.0, 1.0);

                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
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

                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
                            shark_ptr->position[dim] = utils_clamp(
                                    shark_ptr->position[dim] + shark_ptr->speed[dim],
                                    &domain[dim]);
                        }

                        best = OF(shark_ptr->position, cfg.nd, cfg.obj);
                        best_r3 = 0.0;
                        rot_best = (struct RotationBest){ .best = -INFINITY, .best_r3 = 0.0 };
                    }

                    // Rotational local search: each thread evaluates a slice of candidates.
                    #pragma omp for schedule(static) reduction(rot_best: rot_best)
                    for (uint32_t m = 0; m < cfg.rotations; ++m) {
                        double r3 = thread_rand_r(&seed, -1.0, 1.0);
                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
                            candidate[dim] = utils_clamp(
                                    shark_ptr->position[dim] * (1 + r3),
                                    &domain[dim]);
                        }

                        double val = OF(candidate, cfg.nd, cfg.obj);
                        if (val > rot_best.best) {
                            rot_best.best = val;
                            rot_best.best_r3 = r3;
                        }
                    }

                    // Apply best rotation and track global minimum.
                    #pragma omp single
                    {
                        if (rot_best.best > best) {
                            best = rot_best.best;
                            best_r3 = rot_best.best_r3;
                        }

                        if (best_r3 != 0.0) {
                            for (size_t dim = 0; dim < cfg.nd; ++dim) {
                                shark_ptr->position[dim] = utils_clamp(
                                            shark_ptr->position[dim] * (1 + best_r3),
                                            &domain[dim]);
                            }
                        }

                        double cur_min = -best;
                        if (cur_min < best_min) {
                            best_min = cur_min;
                            memcpy(best_pos, shark_ptr->position, cfg.nd * sizeof(double));
                        }
                    }
                } // end stages
            } // end sharks
        } // end parallel

    } // end benchmark

    print_result(best_min, best_pos, cfg.nd);

    // Cleanup
    free(scratch_all);
    free(best_pos);
    sso_sharks_free(sharks, cfg.np);
    free(domain);
    return EXIT_SUCCESS;
}


static error_t parser(int key, char *arg, struct argp_state *state);

static const struct argp_option options[] = {
    {"threads",     't', "int"  , 0, "The number of threads",                    1},
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
        case ARGP_KEY_END:
            return 0;
    }

    return ARGP_ERR_UNKNOWN;
}
