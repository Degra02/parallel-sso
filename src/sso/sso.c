#include "sso.h"
#include "utils.h"

/**
 * @def GRAD_H
 * @brief The quantized step used to compute the gradient.
 */
#define GRAD_H 1e-5

/**
 * @brief Compute the gradient of the objective function given the current
 *        position of the shark.
 * @param pos The position of the shark, must be cfg->nd long.
 * @param cfg The SSO parameters.
 * @param domain The objective function domain.
 * @param grad Where to store the gradient, must be cfg->nd long.
 */
static void compute_gradient(double pos[], const struct SSOConfig *cfg,
                             const struct Interval domain[], double grad[]) {
    for (size_t dim = 0; dim < cfg->nd; ++dim) {
        const double pos_dim = pos[dim];

        // Clamp perturbed points to stay inside the feasible domain.
        double pos_prev = fmax(pos_dim - GRAD_H, domain[dim].start);
        double pos_next = fmin(pos_dim + GRAD_H, domain[dim].end);

        double h_eff = pos_next - pos_prev;
        if (h_eff < 1e-15) {
            // Gradient is not defined on the boundary, treat as 0.
            grad[dim] = 0.0;
        } else {
            grad[dim] = eval_derivative(pos, cfg->nd, cfg->obj, dim);
        }
    }
}

/**
 * @brief Update the speed of the shark considering objective function gradient
 *        and previous speed inertia.
 * @param shark The shark whose speed must be updated.
 * @param cfg The SSO parameters.
 * @param domain The objective function domain.
 * @param grad A support array to store the gradient. Must be cfg->nd long.
 */
void sso_update_speed(struct Shark *shark, const struct SSOConfig *cfg,
                      const struct Interval domain[], double grad[]) {
    double R1 = utils_rand(0, 1); // paper random value
    double R2 = utils_rand(0, 1); // paper random value

    compute_gradient(shark->position, cfg, domain, grad);

    // NOTE: candidate for MP parallelization.
    // TODO: could be interesting to compute also the gradient there, but complications arise.
    for (size_t dim = 0; dim < cfg->nd; ++dim) {
        double v_prev    = shark->speed[dim];
        // gradient based speed: γ·R1·gradient
        double grad_term = cfg->eta * R1 * grad[dim];
        // momentum based speed: α·R2·v_{k-1}
        double mom_term  = cfg->alpha * R2 * v_prev;
        shark->speed[dim] = grad_term + mom_term;

        if (fabs(v_prev) >= 1e-15) {
            // Limit the velocity up to β·v.
            double limit = cfg->beta * v_prev;
            if (fabs(shark->speed[dim]) > fabs(limit)) {
                shark->speed[dim] = limit;
            }
        }
    }
}

/**
 * @brief Shark forward movement based on velocity.
 * @param shark The shark to move.
 * @param cfg The SSO parameters.
 * @param domain The objective function domain.
 */
void sso_move_forward(struct Shark *shark, const struct SSOConfig *cfg,
                      const struct Interval domain[]) {
    for (uint32_t dim = 0; dim < cfg->nd; ++dim) {
        // Assume Δt = 1.
        shark->position[dim] = shark->position[dim] + shark->speed[dim];
    }
    utils_clamp_vec(shark->position, cfg->nd, domain);
}

/**
 * @brief Randomly move to find a better spot, and tell everyone I'm rotating.
 * @param shark The shark that must be teleported.
 * @param cfg The SSO parameters.
 * @domain The objective function domain.
 * @domain candidate A preallocated support array to store candidates. Size cfg->nd.
 * @post shark->pos_score is updated with the value corresponding to shark->position.
 */
void sso_unrotational_search(struct Shark *shark, const struct SSOConfig *cfg,
                             const struct Interval domain[], double candidate[]) {

    // Current best candidate, initialized with the current shark position.
    double best = shark->pos_score = OF(shark->position, cfg->nd, cfg->obj);
    double best_r3 = 0.0;

    for (uint32_t m = 0; m < cfg->rotations; ++m) {
        double r3 = utils_rand(-1, 1);
        // "Rotate" around the shark position.
        for (uint32_t dim = 0; dim < cfg->nd; ++dim) {
            candidate[dim] = shark->position[dim] * (1 + r3);
        }
        utils_clamp_vec(candidate, cfg->nd, domain);

        // Update position if better than the current one.
        double val = OF(candidate, cfg->nd, cfg->obj);
        if (val > best) {
            best = val;
            best_r3 = r3;
        }
    }

    // TODO: evaluate if parallel recomputation can create approximation issues.
    // Update position with the best one.
    // TODO: better to compute only once? How many times we should update it in avg?
    if (best_r3 != 0.0) {
        for (uint32_t dim = 0; dim < cfg->nd; ++dim) {
            shark->position[dim] = shark->position[dim] * (1 + best_r3);
        }
        utils_clamp_vec(shark->position, cfg->nd, domain);
    }
    shark->pos_score = best;
}

/**
 * @brief Perfom a full movement step for every shark in the population.
 * @param sharks The population of sharks.
 * @param cfg The SSO parameters.
 * @param domain The objective function domain.
 * @param scratch A general purpose cfg->nd sized array.
 * @param best_min Contains the current best, updated with the best value after the iteration.
 * @param best_pos Updated with the best position after the iteration.
 */
void sso_perform_step(struct Shark sharks[], const struct SSOConfig *cfg,
                      const struct Interval domain[], double scratch[],
                      double *best_min, double best_pos[]) {
    for (size_t shark = 0; shark < cfg->np; ++shark) {
        // Update shark speed.
        sso_update_speed(&sharks[shark], cfg, domain, scratch);
        // Update shark position with a forward movement.
        sso_move_forward(&sharks[shark], cfg, domain);
        // Try to teleport the shark to a better alternative.
        sso_unrotational_search(&sharks[shark], cfg, domain, scratch);

        // If we have an all-time best value, update the current one.
        double cur_min = - sharks[shark].pos_score;
        if (cur_min < *best_min) {
            *best_min = cur_min;
            memcpy(best_pos, sharks[shark].position, cfg->nd * sizeof(double));
        }

    }
}

/**
 * @brief Allocate and initialize a shark population.
 *        Positions are randomly generated, speeds are set to 0.
 * @param domain The objective function domain.
 * @param cfg The SSO parameters.
 * @return The shark population or NULL in case of exception.
 */
struct Shark *sso_sharks_alloc(const struct Interval *domain,
                               const struct SSOConfig *cfg) {
    struct Shark *sharks = calloc(cfg->np, sizeof(struct Shark));
    if (sharks == NULL) return NULL;

    for (size_t shark = 0; shark < cfg->np; ++shark) {
        // Allocate position and speed arrays.
        sharks[shark].position = calloc(cfg->nd, sizeof(double));
        sharks[shark].speed = calloc(cfg->nd, sizeof(double));
        if (sharks[shark].position == NULL || sharks[shark].speed == NULL) {
            sso_sharks_free(sharks, cfg->np);
            return NULL;
        }

        // Initialize position and speed.
        for (size_t dim = 0; dim < cfg->nd; ++dim) {
            sharks[shark].position[dim] = utils_rand(domain[dim].start, domain[dim].end);
            sharks[shark].speed[dim] = 0.0;
        }
    }

    return sharks;
}

/**
 * @brief Free previously allocated shark population (with sso_sharks_alloc).
 * @param sharks The shark population.
 * @param np The size of the population.
 */
void sso_sharks_free(struct Shark sharks[], size_t np) {
    for (size_t shark = 0; shark < np; ++shark) {
        free(sharks[shark].position);
        free(sharks[shark].speed);
    }

    free(sharks);
}
