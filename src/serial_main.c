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

#include "parse_args.h"
#include "ofuncs.h"
#include "sso.h"

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

    uint32_t show = (nd < 8) ? nd : 8;  /* truncate long position vectors */
    printf("Best x    = [");
    for (uint32_t j = 0; j < show; j++)
        printf(" %.6f", best_pos[j]);
    if (nd > show) printf(" ... (%lu total dims)", nd);
    printf(" ]\n");
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

    /* Seed the PRNG to have reproducible runs. 0 for time-based randomness */
    srand(cfg.seed == 0 ? (unsigned) time(NULL) : (unsigned) cfg.seed);

    print_info(&cfg);

    /* Domain bounds */
    struct Interval *domain = obj_alloc_domain_bounds(cfg.obj, cfg.nd);

    /* Alloc population sharks at random positions in the domain, with 0 speed */
    struct Shark *sharks = sso_sharks_alloc(domain, &cfg);

    /* Scratch array to avoid allocations in loops */
    double *scratch = calloc(cfg.nd, sizeof(double));

    /* The best position found up to date. */
    double *best_pos = calloc(cfg.nd, sizeof(double));

    int ret;
    if (domain == NULL || sharks == NULL || scratch == NULL || best_pos == NULL) {
        perror("Malloc error");
        ret = EXIT_FAILURE;
    } else {
        /* Global best tracked as a minimisation value (f, not OF). */
        double best_min = INFINITY;

        /* Perform k_max movement stages. */
        for (size_t k = 0; k < cfg.k_max; ++k) {
            sso_perform_step(sharks, &cfg, domain, scratch, &best_min, best_pos);

            /* Progress report every 100 stages (and on the first). */
            if (k == 0 || (k + 1) % 100 == 0) {
                printf("Stage %5lu | best f(x) = %.8e\n", k + 1, best_min);
            }
        }

        /* Report final result */
        print_result(best_min, best_pos, cfg.nd);

        ret = EXIT_SUCCESS;
    }

    /* Cleanup */
    free(best_pos);
    free(scratch);
    sso_sharks_free(sharks, cfg.np);
    free(domain);
    return ret;
}
