#!/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=configs.sh
source "$SCRIPT_DIR/configs.sh"

CONFIG="common"

while [[ $# -gt 0 ]]; do
  case $1 in
  -p | --procs)
    N_PROC="$2"
    shift
    shift
    ;;
  -t | --thrds)
    N_THRD="$2"
    shift
    shift
    ;;
  -j | --job)
    JOB_NAME="$2"
    shift
    shift
    ;;
  -e | --exec)
    EXEC="$2"
    shift
    shift
    ;;
  -c | --cpus)
    N_CPUS="$2"
    shift
    shift
    ;;
  -x | --placement)
    PLACE="$2"
    shift
    shift
    ;;
  -g | --config)
    CONFIG="$2"
    shift
    shift
    ;;
  *)
    echo "Unexpected option '$1'"
    exit 1
    ;;
  esac
done

expand_pow2_range() {
  local spec="$1"
  local start end value

  if [[ "$spec" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    start="${BASH_REMATCH[1]}"
    end="${BASH_REMATCH[2]}"
  elif [[ "$spec" =~ ^[0-9]+$ ]]; then
    echo "$spec"
    return 0
  else
    echo "Invalid range '$spec'. Use N or start-end."
    exit 1
  fi

  value=1
  while ((value < start)); do
    value=$((value * 2))
  done

  while ((value <= end)); do
    echo "$value"
    value=$((value * 2))
  done
}

if [ -z "$EXEC" ]; then
  echo "-e|--exec is mandatory"
  exit 1
fi
if [ -z "$JOB_NAME" ]; then
  JOB_NAME="parallel-sso"
fi
if [ -z "$PLACE" ]; then
  PLACE="scatter"
fi
if [ -z "$N_PROC" ]; then
  N_PROC=1
fi
if [ -z "$N_THRD" ]; then
  N_THRD=1
fi
if [ -z "$N_CPUS" ]; then
  N_CPUS=1
fi

# The strategy is the name of the directory holding the template
# (sharks, dim, rot, hybrid). It selects the favorable parameter set.
STRATEGY="$(basename "$(dirname "$EXEC")")"
resolve_config "$CONFIG" "$STRATEGY"

mapfile -t PROC_VALUES < <(expand_pow2_range "$N_PROC")
mapfile -t THRD_VALUES < <(expand_pow2_range "$N_THRD")

for procs in "${PROC_VALUES[@]}"; do
  for thrds in "${THRD_VALUES[@]}"; do
    while [[ $(qstat -u $USER | wc -l) -ge 30 ]]; do
      sleep 10
    done

    if [[ $(basename "$EXEC") == hybrid_sharks* ]]; then
      if ((procs == 64 && thrds == 64)); then
        continue
      fi

      mpi_per_chunk=$((96 / thrds))
      if ((mpi_per_chunk > procs)); then
        mpi_per_chunk=$procs
      fi

      chunks=$(((procs + mpi_per_chunk - 1) / mpi_per_chunk))

      sed \
        -e "s/\${N_THRD}/$thrds/g" \
        -e "s/\${N_PROC}/$procs/g" \
        -e "s/\${JOB_NAME}/$JOB_NAME/g" \
        -e "s/\${N_CHUNKS}/$chunks/g" \
        -e "s/\${N_CPUS_PER_CHUNK}/96/g" \
        -e "s/\${N_MPI_PER_CHUNK}/$mpi_per_chunk/g" \
        -e "s/\${PLACE}/$PLACE/g" \
        -e "s/\${WALLTIME}/$WALLTIME/g" \
        -e "s/\${P}/$P/g" \
        -e "s/\${D}/$D/g" \
        -e "s/\${K}/$K/g" \
        -e "s/\${M}/$M/g" \
        "$EXEC" | cat
    else
      sed \
        -e "s/\${N_THRD}/$thrds/g" \
        -e "s/\${N_PROC}/$procs/g" \
        -e "s/\${JOB_NAME}/$JOB_NAME/g" \
        -e "s/\${N_CPUS}/$N_CPUS/g" \
        -e "s/\${PLACE}/$PLACE/g" \
        -e "s/\${WALLTIME}/$WALLTIME/g" \
        -e "s/\${P}/$P/g" \
        -e "s/\${D}/$D/g" \
        -e "s/\${K}/$K/g" \
        -e "s/\${M}/$M/g" \
        "$EXEC" | cat
    fi

  done
done
