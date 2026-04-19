#ifndef SSO_H_INCLUDED
#define SSO_H_INCLUDED

#include <stdint.h>
#include <stdlib.h>
#include "ofuncs.h"
#include "parse_args.h"

/**
 * @struct Shark
 * @brief Status of one shark in the population.
 */
struct Shark {
    double *position;   /**< Current position of the shark. */
    double *speed;      /**< Current speed of the shark. */
    double pos_score;   /**< The OF(position) value. */
};

void sso_move_forward(struct Shark *shark, const struct SSOConfig *cfg,
                      const struct Interval *domain);
void sso_unrotational_search(struct Shark *shark, const struct SSOConfig *cfg,
                             const struct Interval *domain, double *candidate);
void sso_perform_step(struct Shark *sharks, const struct SSOConfig *cfg,
                      const struct Interval *domain, double *scratch,
                      double *best_min, double *best_pos);
void sso_update_speed(struct Shark *shark, const struct SSOConfig *cfg,
                      const struct Interval *domain, double *grad);

struct Shark *sso_sharks_alloc(const struct Interval *domain,
                               const struct SSOConfig *cfg);
void sso_sharks_free(struct Shark *sharks, size_t np);

#endif /* SSO_H_INCLUDED */
