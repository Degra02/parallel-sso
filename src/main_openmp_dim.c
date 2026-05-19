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
#include <string.h>
#include <stdio.h>
#include <omp.h>

struct Args {
    size_t threads;
};

static const struct argp argp;

/**
 * @brief OpenMP dimensions algorithm entrypoint.
 */
int main(int argc, char *argv[]) {
    // TODO: argp allows to combine different parsers. We should be able to
    // define serial/parallel specific arguments without rewriting everything.
    struct SSOConfig cfg;
    struct Args args = {0};

    if (parse_args_extend(argc, argv, &cfg, &argp, &args) != 0) {
        return EXIT_FAILURE;
    }

    // Seed the PRNG to have reproducible runs. 0 for time-based randomness.
    srand(cfg.seed == 0 ? (unsigned) time(NULL) : (unsigned) cfg.seed);

    print_info(&cfg, "OpenMP Dim");
    printf("threads=%lu ", args.threads);

    // Domain bounds.
    struct Interval *domain = obj_alloc_domain_bounds(cfg.obj, cfg.nd);

    // Alloc population sharks at random positions in the domain, with 0 speed.
    struct Shark *sharks = sso_sharks_alloc(domain, &cfg);

    // Scratch array to avoid allocations in loops.
    double *scratch = calloc(cfg.nd, sizeof(double));

    // The best position found up to date.
    double *best_pos = calloc(cfg.nd, sizeof(double));

    size_t omp_threads = (args.threads == 0) ? omp_get_max_threads() : (size_t) args.threads;
    if (omp_get_max_threads() < omp_threads) {
        fprintf(stderr, "Too many threads %lu/%lu\n", omp_threads, (size_t)omp_get_max_threads());
        return EXIT_FAILURE;
    }

    int ret;
    if (domain == NULL || sharks == NULL || scratch == NULL || best_pos == NULL) {
        perror("Malloc error");
        ret = EXIT_FAILURE;
    } else {
        // Global best. Minimisation problem.
        double best_min = INFINITY;

        double R1 = 0.0;
        double R2 = 0.0;
        double r3 = 0.0;
        double best = 0.0;
        double best_r3 = 0.0;

        BENCHMARK_OPENMP(total_start, "Total OpenMP time") {

            #pragma omp parallel default(none) num_threads(omp_threads) \
                shared(cfg, domain, sharks, scratch, best_pos, best_min, \
                       R1, R2, r3, best, best_r3)
            {
                for (size_t shark = 0; shark < cfg.np; ++shark) {
                    struct Shark *shark_ptr = &sharks[shark];

                    // Perform k_max movement stages.
                    for (size_t k = 0; k < cfg.k_max; ++k) {

                        #pragma omp single
                        {
                            R1 = utils_rand(0, 1);
                            R2 = utils_rand(0, 1);
                        }

                        // Speed update: parallelize over dimensions.
                        #pragma omp for schedule(static)
                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
                            double v_prev = shark_ptr->speed[dim];

                            double derivative = eval_derivative(shark_ptr->position,
                                                        cfg.nd, cfg.obj, dim);
                            // Gradient based speed: γ·R1·gradient.
                            double grad_term = cfg.eta * R1 * derivative;
                            // Momentum based speed: α·R2·v_{k-1}.
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

                        // Position update: parallelize over dimensions.
                        #pragma omp for schedule(static)
                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
                            shark_ptr->position[dim] = utils_clamp(
                                    shark_ptr->position[dim] + shark_ptr->speed[dim],
                                    &domain[dim]);
                        }

                        // Try to teleport the shark to a better alternative.
                        #pragma omp single
                        {
                            best = OF(shark_ptr->position, cfg.nd, cfg.obj);
                            best_r3 = 0.0;
                        }

                        for (uint32_t m = 0; m < cfg.rotations; ++m) {
                            #pragma omp single
                            {
                                r3 = utils_rand(-1, 1);
                            }

                            // Rotate around the shark position.
                            #pragma omp for schedule(static)
                            for (size_t dim = 0; dim < cfg.nd; ++dim) {
                                scratch[dim] = utils_clamp(
                                            shark_ptr->position[dim] * (1 + r3),
                                            &domain[dim]);
                            }

                            // Update position if better than the current one.
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

                        // If we have an all-time best value, update the current one.
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
        }

        // Print final result.
        print_result(best_min, best_pos, cfg.nd);

        ret = EXIT_SUCCESS;
    }

    // Cleanup.
    free(best_pos);
    free(scratch);
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
    struct Args *args = state->input;

    switch (key) {
        case 't':
            RET_PARSE_ULL(args, threads, arg);
        case ARGP_KEY_END:
            return 0;
    }

    return ARGP_ERR_UNKNOWN;
}
