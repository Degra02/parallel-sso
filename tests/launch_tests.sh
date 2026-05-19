#!/bin/env bash

while [[ $# -gt 0 ]]; do
  case $1 in
  -p | --procs)
    N_PROC="$2"
    shift # past argument
    shift # past value
    ;;
  -t | --thrds)
    N_THRD="$2"
    shift # past argument
    shift # past value
    ;;
  -j | --job)
    JOB_NAME="$2"
    shift # past argument
    shift # past value
    ;;
  -e | --exec)
    EXEC="$2"
    shift # past argument
    shift # past value
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
    start=1
    end="$spec"
  else
    echo "Invalid range '$spec'. Use N or start-end (for example 2-8)."
    exit 1
  fi

  if ((start < 1 || end < start)); then
    echo "Invalid range '$spec'. Ensure 1 <= start <= end."
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

# if [ -z "$N_PROC" ]; then
#   echo "-p|--procs is mandatory"
#   exit 1
# fi
# if [ -z "$N_THRD" ]; then
#   echo "-t|--thrds is mandatory"
#   exit 1
# fi
# if [ -z "$JOB_NAME" ]; then
#   echo "-j|--job is mandatory"
#   exit 1
# fi

mapfile -t PROC_VALUES < <(expand_pow2_range "$N_PROC")
mapfile -t THRD_VALUES < <(expand_pow2_range "$N_THRD")

for procs in "${PROC_VALUES[@]}"; do
  for thrds in "${THRD_VALUES[@]}"; do
    total_cpus=$procs
    if ((thrds > total_cpus)); then
      total_cpus=$thrds
    fi
    while [[ $(qstat -u $USER | wc -l) -ge 30 ]]; do
      sleep 10
    done
    sed \
      -e "s/\${N_THRD}/$thrds/g" \
      -e "s/\${N_PROC}/$procs/g" \
      -e "s/\${JOB_NAME}/$JOB_NAME/g" \
      -e "s/\${N_CPU}/$total_cpus/g" \
      "$EXEC" | cat
  done
done
