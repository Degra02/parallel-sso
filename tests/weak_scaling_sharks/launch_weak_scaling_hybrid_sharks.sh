#!/usr/bin/env bash

set -euo pipefail

BASE_POPULATION=1000
CORES_PER_NODE=96
MAX_QUEUED_JOBS=30
PLACE="scatter"
JOB_NAME="weak-hybrid-sharks"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
template="${script_dir}/weak_scaling_hybrid_sharks"
raw_dir="${repo_root}/benchmarks/raw/weak_scaling_hybrid_sharks"

procs_values=(1 2 4 8 16 32 64)
thread_values=(1 2 4 8 16 32 64)

mkdir -p "${raw_dir}"

cd "${repo_root}"

for procs in "${procs_values[@]}"; do
  : > "${raw_dir}/weak_scaling_hybrid_sharks_${procs}.txt"
  rm -f "${raw_dir}/weak_scaling_hybrid_sharks_${procs}.txt.lock"
done

queued_job_count() {
  qselect -u "${USER}" | wc -l
}

for procs in "${procs_values[@]}"; do
  raw_file="${raw_dir}/weak_scaling_hybrid_sharks_${procs}.txt"

  for threads in "${thread_values[@]}"; do
    if ((procs == 64 && threads == 64)); then
      continue
    fi

    workers=$((procs * threads))
    population=$((BASE_POPULATION * workers))

    mpi_per_chunk=$((CORES_PER_NODE / threads))
    if ((mpi_per_chunk > procs)); then
      mpi_per_chunk=${procs}
    fi
    chunks=$(((procs + mpi_per_chunk - 1) / mpi_per_chunk))

    pbs_job=$(
      sed \
        -e "s/\${N_THRD}/${threads}/g" \
        -e "s/\${N_PROC}/${procs}/g" \
        -e "s/\${N_WORKERS}/${workers}/g" \
        -e "s/\${POPULATION}/${population}/g" \
        -e "s/\${JOB_NAME}/${JOB_NAME}/g" \
        -e "s/\${N_CHUNKS}/${chunks}/g" \
        -e "s/\${N_MPI_PER_CHUNK}/${mpi_per_chunk}/g" \
        -e "s/\${PLACE}/${PLACE}/g" \
        "${template}"
    )

    while (( $(queued_job_count) >= MAX_QUEUED_JOBS )); do
      sleep 10
    done

    printf '%s\n' "${pbs_job}" | qsub
  done
done
