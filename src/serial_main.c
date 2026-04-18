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

#include <math.h>
#include <time.h>


/*
 *  4 – Numerical gradient of OF  (Eq. 6)
 *
 *  Central-difference approximation:
 *    ∂OF/∂xj ≈ [ OF(x + h·ej) - OF(x - h·ej) ] / (2h)
 *
 *  scratch_xp and scratch_xm are caller-allocated ND-sized buffers
 *  (avoids repeated malloc/free inside the inner loop).
 * ═══════════════════════════════════════════════════════════════════════ */

#define GRAD_H 1e-5   /* finite-difference step size */

static void compute_gradient(const double *x, uint32_t nd,
                              ObjectiveFunction obj,
                              const double *lb, const double *ub,
                              double *grad,
                              double *scratch_xp, double *scratch_xm) {
    for (uint32_t j = 0; j < nd; j++) {
        /* Copy current position into scratch buffers, then perturb dim j. */
        memcpy(scratch_xp, x, nd * sizeof(double));
        memcpy(scratch_xm, x, nd * sizeof(double));

        /* Clamp perturbed points to stay inside the feasible domain. */
        scratch_xp[j] = fmin(x[j] + GRAD_H, ub[j]);
        scratch_xm[j] = fmax(x[j] - GRAD_H, lb[j]);

        double h_eff = scratch_xp[j] - scratch_xm[j];
        if (h_eff < 1e-15)
            grad[j] = 0.0;  /* boundary: gradient undefined, treat as 0 */
        else
            grad[j] = (OF(scratch_xp, nd, obj) - OF(scratch_xm, nd, obj))
                      / h_eff;
    }
}

// 5 Utility helpers

//Clamp every dimension of x into [lb[j], ub[j]]. */
static void clamp_pos(double *x, uint32_t nd,
                      const double *lb, const double *ub) {
    for (uint32_t j = 0; j < nd; j++) {
        if (x[j] < lb[j]) x[j] = lb[j];
        if (x[j] > ub[j]) x[j] = ub[j];
    }
}

static inline double rand01(void) {
    return (double)rand() / ((double)RAND_MAX + 1.0);
}


static inline double randm11(void) {
    return 2.0 * rand01() - 1.0;
}

// 6  Main
int main(int argc, char *argv[]) {
    struct SSOConfig cfg;
    if (parse_args(argc, argv, &cfg) != 0) {
        return 1;
    }

    /* Seed the PRNG; 0 → time-based non-reproducible run. */
    srand(cfg.seed == 0 ? (unsigned)time(NULL) : (unsigned)cfg.seed);

    /* Convenient local aliases */
    const uint32_t NP   = cfg.np;     /* population size                  */
    const uint32_t ND   = cfg.nd;     /* number of decision variables      */
    const uint32_t KMAX = cfg.k_max;  /* maximum stages                   */
    const uint32_t M    = cfg.rotations;      /* rotational-search probe points    */
    const double   nk   = cfg.mu;   /* γ – gradient scaling factor       */
    const double   ak   = cfg.alpha;   /* α – momentum (inertia) rate       */
    const double   bk   = cfg.beta;   /* β – velocity limiter ratio        */
    const ObjectiveFunction obj = cfg.obj;

    printf("=== SSO Serial  ===\n");
    printf("  NP=%u  ND=%u  k_max=%u  M=%u\n",   NP, ND, KMAX, M);
    printf("  gamma=%.3f  alpha=%.3f  beta=%.3f\n", nk, ak, bk);
    printf("  objective=%d  seed=%llu\n\n",
           (int)obj, (unsigned long long)cfg.seed);

    /* ── Build per-dimension bounds arrays ── */
    double *lb = malloc(ND * sizeof(double));
    double *ub = malloc(ND * sizeof(double));
    if (!lb || !ub) { fprintf(stderr, "malloc failed\n"); return 1; }
    for (uint32_t j = 0; j < ND; j++) {
        lb[j] = domain_lb(obj);
        ub[j] = domain_ub(obj);
    }

    /* Allocate all working arrays up front to avoid repeated malloc in
     * the inner loop.
     *
     *   X[NP][ND]       – current shark positions  (Eq. 2)
     *   V[NP][ND]       – current shark velocities (Eq. 4)
     *   Y[ND]           – candidate after forward step (Eq. 9)
     *   Z[M][ND]        – rotational-search probes (Eq. 10)
     *   grad[ND]        – ∇OF at current position  (Eq. 6)
     *   scratch_xp/xm   – scratch buffers for finite differences
     *   best_x[ND]      – best position found so far  */
    double **X        = malloc(NP * sizeof(double *));
    double **V        = malloc(NP * sizeof(double *));
    double  *Y        = malloc(ND * sizeof(double));
    double **Z        = malloc(M  * sizeof(double *));
    double  *grad     = malloc(ND * sizeof(double));
    double  *scratch_xp = malloc(ND * sizeof(double));
    double  *scratch_xm = malloc(ND * sizeof(double));
    double  *best_x   = malloc(ND * sizeof(double));

    if (!X || !V || !Y || !Z || !grad || !scratch_xp || !scratch_xm || !best_x) {
        fprintf(stderr, "malloc failed\n"); return 1;
    }

    for (uint32_t i = 0; i < NP; i++) {
        X[i] = malloc(ND * sizeof(double));
        V[i] = malloc(ND * sizeof(double));
        if (!X[i] || !V[i]) { fprintf(stderr, "malloc failed\n"); return 1; }
    }
    for (uint32_t m = 0; m < M; m++) {
        Z[m] = malloc(ND * sizeof(double));
        if (!Z[m]) { fprintf(stderr, "malloc failed\n"); return 1; }
    }

    /* Step 1 – Initialisation  ( 3.1, Eq. 1-2)
     *     * Each shark is placed at a uniformly random position
     * inside the feasible domain.  Initial velocities are
     * set to zero */

    for (uint32_t i = 0; i < NP; i++) {
        for (uint32_t j = 0; j < ND; j++) {
            X[i][j] = lb[j] + rand01() * (ub[j] - lb[j]);
            V[i][j] = 0.0;
        }
    }

    /* Global best tracked as a minimisation value (f, not OF). */
    double best_min = 1e300;

    /*
     * Step 2 – Evolution loop
    * Stages k = 0 … KMAX-1
     */
    for (uint32_t k = 0; k < KMAX; k++) {

        for (uint32_t i = 0; i < NP; i++) {

            /* A. Numerical gradient  (Eq. 6)
             *
             * grad[j] = ∂OF/∂xj | X^k_i
             *
             * Points in the direction of steepest ascent of OF, i.e.
             * toward stronger odour / lower f.                          */
            compute_gradient(X[i], ND, obj, lb, ub, grad,
                             scratch_xp, scratch_xm);

            /*B. Velocity update  (Eq. 7-8)
             *
             * Unconstrained candidate (Eq. 7):
             *   candidate_j = γ·R1·grad_j  +  α·R2·v^{k-1}_{i,j}
             *
             * Velocity limiter (Eq. 8):
             *   |v^k_{i,j}| = min( |candidate_j|,  |β·v^{k-1}_{i,j}| )
             *   sign         = sign of the chosen term
             *
             * Physical motivation: a shark can accelerate at most β×
             * its previous speed in a single stage (β = 4 from the
             * 80 km/h attack / 20 km/h cruise ratio, 3.2).
             * Edge case: when v^{k-1} ≈ 0 (first stage or stall) the
             * limiter β·v = 0 would freeze the shark.  We skip the
             * limiter in that case and accept the full candidate.       */
            double R1 = rand01();   /* one scalar per (k, i) Eq. 5   */
            double R2 = rand01();   /* idem for momentum term           */

            for (uint32_t j = 0; j < ND; j++) {
                double v_prev    = V[i][j];
                double grad_term = nk * R1 * grad[j];
                double mom_term  = ak * R2 * v_prev;
                double candidate = grad_term + mom_term;

                if (fabs(v_prev) < 1e-15) {
                    /* Limiter is inapplicable; use gradient directly. */
                    V[i][j] = candidate;
                } else {
                    double limiter  = bk * v_prev;
                    double abs_cand = fabs(candidate);
                    double abs_lim  = fabs(limiter);
                    /* Pick the term with smaller magnitude; preserve its
                     * sign (the sign of the selected term in min()).    */
                    V[i][j] = (abs_cand <= abs_lim) ? candidate : limiter;
                }
            }

            /* ── C. Forward movement  (Eq. 9, Δt = 1)
             *
             * Y^{k+1}_i = X^k_i  +  V^k_i · Δt
             *
             * Clamp to the feasible domain after stepping.              */
            for (uint32_t j = 0; j < ND; j++)
                Y[j] = X[i][j] + V[i][j];
            clamp_pos(Y, ND, lb, ub);

            /* ── D. Rotational local search  (Eq. 10)
             *
             * Shark performs a rotational sweep around Y to look for
             * stronger odour nearby (models the natural circular movement .
             *
             * M probe points:
             *   Z^{k+1,m}_i  =  Y  +  R3·Y  =  Y·(1 + R3)
             * where R3 ~ Uniform[-1, 1] is one scalar per probe m.
             * When R3 = 0 the probe coincides with Y.                  */
            for (uint32_t m = 0; m < M; m++) {
                double R3 = randm11();
                for (uint32_t j = 0; j < ND; j++)
                    Z[m][j] = Y[j] + R3 * Y[j];   /* = Y[j] * (1 + R3) */
                clamp_pos(Z[m], ND, lb, ub);
            }

            /* E. Selection  (Eq. 11)
             *
             * X^{k+1}_i = argmax{ OF(Y), OF(Z^1), …, OF(Z^M) }
             *
             * The candidate with the highest OF value (strongest odour)
             * becomes the shark's next position.                        */
            double  best_of  = OF(Y, ND, obj);
            double *best_pos = Y;

            for (uint32_t m = 0; m < M; m++) {
                double of_val = OF(Z[m], ND, obj);
                if (of_val > best_of) {
                    best_of  = of_val;
                    best_pos = Z[m];
                }
            }

            /* Commit the winner as the new position. */
            memcpy(X[i], best_pos, ND * sizeof(double));

            /* F. Global best tracking value: f(x) = -OF(x). */
            double cur_min = -best_of;
            if (cur_min < best_min) {
                best_min = cur_min;
                memcpy(best_x, X[i], ND * sizeof(double));
            }

        }

        /* Progress report every 100 stages (and on the first). */
        if (k == 0 || (k + 1) % 100 == 0)
            printf("Stage %5u | best f(x) = %.8e\n", k + 1, best_min);

    }

    /* Step 3 – Report final result*/

    printf("\n=== Final Result ===\n");
    printf("Best f(x) = %.10e\n", best_min);

    uint32_t show = (ND < 8) ? ND : 8;  /* truncate long position vectors */
    printf("Best x    = [");
    for (uint32_t j = 0; j < show; j++)
        printf(" %.6f", best_x[j]);
    if (ND > show) printf(" ... (%u total dims)", ND);
    printf(" ]\n");

    /*  Free all heap memory */
    for (uint32_t i = 0; i < NP; i++) { free(X[i]); free(V[i]); }
    for (uint32_t m = 0; m < M; m++)   free(Z[m]);
    free(X); free(V); free(Y); free(Z);
    free(grad); free(scratch_xp); free(scratch_xm);
    free(best_x); free(lb); free(ub);

    return 0;
}
