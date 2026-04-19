#include "sso.h"
#include "utils.h"

#define GRAD_H 1e-5   /* finite-difference step size */
static void compute_gradient(struct Shark *shark, const struct SSOConfig *cfg,
                             const struct Interval *domain, double *grad) {
    for (size_t dim = 0; dim < cfg->nd; ++dim) {
        const double pos_dim = shark->position[dim];

        /* Clamp perturbed points to stay inside the feasible domain. */
        double pos_prev = fmax(pos_dim - GRAD_H, domain[dim].start);
        double pos_next = fmin(pos_dim + GRAD_H, domain[dim].end);

        double h_eff = pos_next - pos_prev;
        if (h_eff < 1e-15) {
            grad[dim] = 0.0;  /* boundary: gradient undefined, treat as 0 */
        } else {
            // TODO: not an optimal approach if parallelized.
            // Do we need to parallelize this? Otherwise we need to allocate
            // multiple arrays and copy positions.
            shark->position[dim] = pos_prev;
            double of_prev = OF(shark->position, cfg->nd, cfg->obj);
            shark->position[dim] = pos_next;
            double of_next = OF(shark->position, cfg->nd, cfg->obj);
            grad[dim] = (of_next - of_prev) / h_eff;
            // Restore position.
            shark->position[dim] = pos_dim;
        }
    }
}

void sso_update_speed(struct Shark *shark, const struct SSOConfig *cfg,
                      const struct Interval *domain, double *grad) {
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
    double R1 = utils_rand(0, 1);   /* one scalar per (k, i) Eq. 5   */
    double R2 = utils_rand(0, 1);   /* idem for momentum term           */

    /* A. Numerical gradient  (Eq. 6)
     *
     * grad[j] = ∂OF/∂xj | X^k_i
     *
     * Points in the direction of steepest ascent of OF, i.e.
     * toward stronger odour / lower f.                          */
    compute_gradient(shark, cfg, domain, grad);

    for (size_t dim = 0; dim < cfg->nd; ++dim) {
        double v_prev    = shark->speed[dim];
        double grad_term = cfg->eta * R1 * grad[dim];
        double mom_term  = cfg->alpha * R2 * v_prev;
        double candidate = grad_term + mom_term;

        if (fabs(v_prev) < 1e-15) {
            /* Limiter is inapplicable; use gradient directly. */
            shark->speed[dim] = candidate;
        } else {
            double limiter = cfg->beta * v_prev;
            /* Pick the term with smaller magnitude; preserve its
             * sign (the sign of the selected term in min()).    */
            if (fabs(candidate) <= fabs(limiter)) {
                shark->speed[dim] = candidate;
            } else {
                shark->speed[dim] = limiter;
            }
        }
    }
}

void sso_move_forward(struct Shark *shark, const struct SSOConfig *cfg,
                      const struct Interval *domain) {
    /* ── C. Forward movement  (Eq. 9, Δt = 1)
     *
     * Y^{k+1}_i = X^k_i  +  V^k_i · Δt
     *
     * Clamp to the feasible domain after stepping.              */
    for (uint32_t dim = 0; dim < cfg->nd; ++dim) {
        shark->position[dim] = shark->position[dim] + shark->speed[dim];
    }
    utils_clamp_vec(shark->position, cfg->nd, domain);
    shark->pos_score = OF(shark->position, cfg->nd, cfg->obj);
}


void sso_unrotational_search(struct Shark *shark, const struct SSOConfig *cfg,
                             const struct Interval *domain, double *candidate) {
    double best = shark->pos_score;
    double best_r3 = 0.0;

    /* ── D. Rotational local search  (Eq. 10)
     *
     * Shark performs a rotational sweep around Y to look for
     * stronger odour nearby (models the natural circular movement).
     *
     * M probe points:
     *   Z^{k+1,m}_i  =  Y  +  R3·Y  =  Y·(1 + R3)
     * where R3 ~ Uniform[-1, 1] is one scalar per probe m.
     * When R3 = 0 the probe coincides with Y.                  */
    for (uint32_t m = 0; m < cfg->rotations; ++m) {
        double r3 = utils_rand(-1, 1);
        for (uint32_t dim = 0; dim < cfg->nd; ++dim) {
            // NOTE: clearly a rotational movement.
            candidate[dim] = shark->position[dim] * (1 + r3);   /* = Y[j] * (1 + R3) */
        }
        utils_clamp_vec(candidate, cfg->nd, domain);

        double val = OF(candidate, cfg->nd, cfg->obj);
        if (val > best) {
            best = val;
            best_r3 = r3;
        }
    }

    // TODO: evaluate if parallel recomputation can create approximation issues.
    // Update position with the best one.
    if (best_r3 != 0.0) {
        for (uint32_t dim = 0; dim < cfg->nd; ++dim) {
            shark->position[dim] = shark->position[dim] * (1 + best_r3);
        }
        utils_clamp_vec(shark->position, cfg->nd, domain);
    }
    shark->pos_score = best;
}


void sso_perform_step(struct Shark *sharks, const struct SSOConfig *cfg,
                      const struct Interval *domain, double *scratch,
                      double *best_min, double *best_pos) {
    for (size_t shark = 0; shark < cfg->np; ++shark) {
        sso_update_speed(&sharks[shark], cfg, domain, scratch);

        sso_move_forward(&sharks[shark], cfg, domain);

        sso_unrotational_search(&sharks[shark], cfg, domain, scratch);

        /* F. Global best tracking value: f(x) = -OF(x). */
        double cur_min = - sharks[shark].pos_score;
        if (cur_min < *best_min) {
            *best_min = cur_min;
            memcpy(best_pos, sharks[shark].position, cfg->nd * sizeof(double));
        }

    }
}

struct Shark *sso_sharks_alloc(const struct Interval *domain,
                               const struct SSOConfig *cfg) {
    struct Shark *sharks = calloc(cfg->np, sizeof(struct Shark));
    if (sharks == NULL) return NULL;

    for (size_t shark = 0; shark < cfg->np; ++shark) {
        sharks[shark].position = calloc(cfg->nd, sizeof(double));
        if (sharks[shark].position == NULL) {
            sso_sharks_free(sharks, shark);
            return NULL;
        }
        sharks[shark].speed = calloc(cfg->nd, sizeof(double));
        if (sharks[shark].speed == NULL) {
            free(sharks[shark].speed);
            sso_sharks_free(sharks, shark);
            return NULL;
        }

        for (size_t dim = 0; dim < cfg->nd; ++dim) {
            sharks[shark].position[dim] = utils_rand(domain[dim].start, domain[dim].end);
            sharks[shark].speed[dim] = 0.0;
        }

        sharks[shark].pos_score = OF(sharks[shark].position, cfg->nd, cfg->obj);
    }

    return sharks;
}

void sso_sharks_free(struct Shark *sharks, size_t np) {
    for (size_t shark = 0; shark < np; ++shark) {
        free(sharks[shark].position);
        free(sharks[shark].speed);
    }

    free(sharks);
}
