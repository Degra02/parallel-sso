#!/bin/env bash
#
# Parameter sets for the benchmark runs, consumed by launch_tests.sh.
#
#   common : fixed configuration used for the fair cross-strategy comparison.
#            The same parameters are used for every strategy, so the overlaid
#            MPI/OpenMP/hybrid plots are apples-to-apples.
#
#   fav    : strategy-specific "favorable" configuration. Each strategy gets a
#            parameter set whose parallel axis (np / nd / M) is much larger than
#            the maximum worker count, and whose per-unit work amortizes the
#            synchronization/communication cost. Each set has its own T(1).
#

# --- Fixed / common configuration (shared by all strategies) ---
COMMON_P=1000
COMMON_D=200
COMMON_K=1000
COMMON_M=50
COMMON_WALLTIME=0:10:00

# --- Favorable: shark-level (big population so every core owns sharks) ---
FAV_sharks_P=8192
FAV_sharks_D=100
FAV_sharks_K=250
FAV_sharks_M=50
FAV_sharks_WALLTIME=0:10:00

# --- Favorable: dimension-level (big nd, few sharks; M small since the ---
# --- rotation search is serial in the dim variant) ---
FAV_dim_P=5
FAV_dim_D=100000000
FAV_dim_K=5
FAV_dim_M=0
FAV_dim_WALLTIME=0:20:00

# --- Favorable: rotation-level (big M, few sharks) ---
FAV_rot_P=16
FAV_rot_D=200
FAV_rot_K=100
FAV_rot_M=20000
FAV_rot_WALLTIME=0:15:00

# --- Favorable: hybrid (combined) decomposition: reuse the shark set ---
FAV_hybrid_P=8192
FAV_hybrid_D=100
FAV_hybrid_K=250
FAV_hybrid_M=50
FAV_hybrid_WALLTIME=0:10:00

# resolve_config <common|fav> <strategy>
# Exports P, D, K, M, WALLTIME for the requested config/strategy.
resolve_config() {
  local cfg="$1" strat="$2"

  if [[ "$cfg" == "common" ]]; then
    P=$COMMON_P; D=$COMMON_D; K=$COMMON_K; M=$COMMON_M; WALLTIME=$COMMON_WALLTIME
  elif [[ "$cfg" == "fav" ]]; then
    local pv="FAV_${strat}_P" dv="FAV_${strat}_D" kv="FAV_${strat}_K"
    local mv="FAV_${strat}_M" wv="FAV_${strat}_WALLTIME"
    P=${!pv}; D=${!dv}; K=${!kv}; M=${!mv}; WALLTIME=${!wv}
    if [[ -z "$P" ]]; then
      echo "No favorable config defined for strategy '$strat'" >&2
      exit 1
    fi
  else
    echo "Unknown config '$cfg' (use 'common' or 'fav')" >&2
    exit 1
  fi
}
