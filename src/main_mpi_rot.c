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
        print_info(&cfg, "MPI Dim");
        printf("procs=%d ", size);
    }

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
        size_t remainder = cfg.rotations % size;
        size_t start_rot = cfg.rotations / size * rank + (rank < remainder ? rank : remainder);
        size_t end_rot = start_rot + cfg.rotations / size + (rank < remainder ? 1 : 0);

        MPI_Barrier(MPI_COMM_WORLD);

        BENCHMARK(total_start, "Total time") {

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
                    struct { double best,  best_r3; } buf;
                    buf.best = OF(shark_ptr->position,
                                     cfg.nd, cfg.obj);
                    buf.best_r3 = 0.0;

                    for (uint32_t m = start_rot; m < end_rot; ++m) {
                        // TODO: random issues: every process computes the same rotations.
                        double r3 = utils_rand(-1, 1);
                        // "Rotate" around the shark position.
                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
                            candidate[dim] = utils_clamp(
                                        shark_ptr->position[dim] * (1 + r3),
                                        &domain[dim]);
                        }

                        // Update position if better than the current one.
                        double val = OF(candidate, cfg.nd, cfg.obj);
                        if (val > buf.best) {
                            buf.best = val;
                            buf.best_r3 = r3;
                        }
                    }

                    MPI_Allreduce(MPI_IN_PLACE, &buf, 1, MPI_2DOUBLE_PRECISION, MPI_MAXLOC, MPI_COMM_WORLD);

                    if (buf.best_r3 != 0.0) {
                        for (size_t dim = 0; dim < cfg.nd; ++dim) {
                            shark_ptr->position[dim] = utils_clamp(
                                        shark_ptr->position[dim] * (1 + buf.best_r3),
                                        &domain[dim]);
                        }
                    }

                    // If we have an all-time best value, update the current one.
                    double cur_min = -buf.best;
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
    free(best_pos);
    free(scratch);
    sso_sharks_free(sharks, cfg.np);
    free(domain);
    return ret;
}
