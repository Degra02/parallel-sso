#!/usr/bin/env bash

set -euo pipefail

values=(1 2 4 8 16 32 64)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
output_dir="${repo_root}/benchmarks/raw/weak_scaling_sharks"

cd "${repo_root}"

mkdir -p "${output_dir}"
rm -f "${output_dir}"/*.txt "${output_dir}"/*.lock
touch \
  "${output_dir}/weak_scaling_openmp_sharks.txt" \
  "${output_dir}/weak_scaling_mpi_sharks.txt"

for n in "${values[@]}"; do
  qsub \
    -N "ws_omp_${n}" \
    -v "N_THREADS=${n}" \
    -l "select=1:ncpus=${n}:mem=2gb" \
    "${script_dir}/weak_scaling_openmp_sharks"
done

for n in "${values[@]}"; do
  mpi_memory="2gb"
  if ((n == 64)); then
    mpi_memory="8gb"
  fi

  qsub \
    -N "ws_mpi_${n}" \
    -v "N_PROCS=${n}" \
    -l "select=1:ncpus=${n}:mpiprocs=${n}:mem=${mpi_memory}" \
    "${script_dir}/weak_scaling_mpi_sharks"
done
