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

#include <math.h>
#include <time.h>
#include <stdlib.h>

static void print_info(const struct SSOConfig *cfg) {
    printf("=== SSO Serial  ===\n");
    printf("NP=%lu\t\tND=%lu\t\tk_max=%lu\tM=%lu\n",
           cfg->np, cfg->nd, cfg->k_max, cfg->rotations);
    printf("eta=%.3f\talpha=%.3f\tbeta=%.3f\n",
           cfg->eta, cfg->alpha, cfg->beta);
    printf("objective=%d\tseed=%lu\n\n", cfg->obj, cfg->seed);
}

static void print_result(double best_min, const double *best_pos, size_t nd) {
    printf("\n=== Final Result ===\n");
    printf("Best f(x) = %.10e\n", best_min);

    size_t count_per_row = 8;
    printf("Best x    = [");
    for (uint32_t j = 0; j < nd; j++) {
        if (j % count_per_row == 0) printf("\n");
        printf(" %9.6f", best_pos[j]);
    }
    printf("\n]\n");
}

/**
 * @brief Serial algorithm entrypoint.
 */
int main(int argc, char *argv[]) {
    // TODO: argp allows to combine different parsers. We should be able to
    // define serial/parallel specific arguments without rewriting everything.
    struct SSOConfig cfg;
    if (parse_args(argc, argv, &cfg) != 0) {
        return EXIT_FAILURE;
    }

    // Seed the PRNG to have reproducible runs. 0 for time-based randomness.
    srand(cfg.seed == 0 ? (unsigned) time(NULL) : (unsigned) cfg.seed);

    print_info(&cfg);

    // Domain bounds.
    struct Interval *domain = obj_alloc_domain_bounds(cfg.obj, cfg.nd);

    // Alloc population sharks at random positions in the domain, with 0 speed.
    struct Shark *sharks = sso_sharks_alloc(domain, &cfg);

    // Scratch array to avoid allocations in loops.
    double *scratch = calloc(cfg.nd, sizeof(double));

    // The best position found up to date.
    double *best_pos = calloc(cfg.nd, sizeof(double));

    int ret;
    if (domain == NULL || sharks == NULL || scratch == NULL || best_pos == NULL) {
        perror("Malloc error");
        ret = EXIT_FAILURE;
    } else {
        // Global best. Minimisation problem.
        double best_min = INFINITY;

        for (size_t shark = 0; shark < cfg.np; ++shark) {
            struct Shark *shark_ptr = &sharks[shark];

            // Perform k_max movement stages.
            for (size_t k = 0; k < cfg.k_max; ++k) {
                // Update shark speed.
                double R1 = utils_rand(0, 1); // paper random value
                double R2 = utils_rand(0, 1); // paper random value

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
                double best = OF(shark_ptr->position,
                                                    cfg.nd, cfg.obj);
                double best_r3 = 0.0;

                for (uint32_t m = 0; m < cfg.rotations; ++m) {
                    double r3 = utils_rand(-1, 1);
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
                if (cur_min < best_min) {
                    best_min = cur_min;
                    memcpy(best_pos, shark_ptr->position, cfg.nd * sizeof(double));
                }
            }
        }

        // Print final result.
        print_result(best_min, best_pos, cfg.nd);

        ret = EXIT_SUCCESS;
    }

    // Cleanup
    free(best_pos);
    free(scratch);
    sso_sharks_free(sharks, cfg.np);
    free(domain);
    return ret;
}
