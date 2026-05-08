#ifndef SSO_H_INCLUDED
#define SSO_H_INCLUDED

#include "ofuncs.h"
#include "parse_args.h"
#include <stdint.h>
#include <stdlib.h>

/**
 * @struct Shark
 * @brief Status of one shark in the population.
 */
struct Shark {
  double *position; /**< Current position of the shark. */
  double *speed;    /**< Current speed of the shark. */
  double pos_score; /**< The OF(position) value. */
};

/**
 * @brief Update the speed of the shark considering objective function gradient
 *        and previous speed inertia.
 * @param shark The shark whose speed must be updated.
 * @param cfg The SSO parameters.
 * @param domain The objective function domain.
 * @param grad A support array to store the gradient. Must be cfg->nd long.
 */
void sso_update_speed(struct Shark *shark, const struct SSOConfig *cfg,
                      const struct Interval *domain, double *grad);

/**
 * @brief Shark forward movement based on velocity.
 * @param shark The shark to move.
 * @param cfg The SSO parameters.
 * @param domain The objective function domain.
 */
void sso_move_forward(struct Shark *shark, const struct SSOConfig *cfg,
                      const struct Interval *domain);

/**
 * @brief Randomly move to find a better spot, and tell everyone I'm rotating.
 * @param shark The shark that must be teleported.
 * @param cfg The SSO parameters.
 * @domain The objective function domain.
 * @domain candidate A preallocated support array to store candidates. Size
 * cfg->nd.
 * @post shark->pos_score is updated with the value corresponding to
 * shark->position.
 */
void sso_unrotational_search(struct Shark *shark, const struct SSOConfig *cfg,
                             const struct Interval *domain, double *candidate);

/**
 * @brief Perfom a full movement step for every shark in the population.
 * @param sharks The population of sharks.
 * @param cfg The SSO parameters.
 * @param domain The objective function domain.
 * @param scratch A general purpose cfg->nd sized array.
 * @param best_min Contains the current best, updated with the best value after
 * the iteration.
 * @param best_pos Updated with the best position after the iteration.
 */
void sso_perform_step(struct Shark *sharks, const struct SSOConfig *cfg,
                      const struct Interval *domain, double *scratch,
                      double *best_min, double *best_pos);

/**
 * @brief Allocate and initialize a shark population.
 *        Positions are randomly generated, speeds are set to 0.
 * @param domain The objective function domain.
 * @param cfg The SSO parameters.
 * @return The shark population or NULL in case of exception.
 */
struct Shark *sso_sharks_alloc(const struct Interval *domain,
                               const struct SSOConfig *cfg);

/**
 * @brief Free previously allocated shark population (with sso_sharks_alloc).
 * @param sharks The shark population.
 * @param np The size of the population.
 */
void sso_sharks_free(struct Shark *sharks, size_t np);

#endif /* SSO_H_INCLUDED */
