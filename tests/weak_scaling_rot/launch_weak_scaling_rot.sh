#!/usr/bin/env bash

set -euo pipefail

values=(1 2 4 8 16 32 64)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
output_dir="${repo_root}/benchmarks/raw/weak_scaling_rot"

cd "${repo_root}"

mkdir -p "${output_dir}"
rm -f "${output_dir}"/*.txt "${output_dir}"/*.lock
touch \
  "${output_dir}/weak_scaling_openmp_rot.txt" \
  "${output_dir}/weak_scaling_mpi_rot.txt"

for n in "${values[@]}"; do
  qsub \
    -N "ws_omp_rot_${n}" \
    -v "N_THREADS=${n}" \
    -l "select=1:ncpus=${n}:mem=2gb" \
    "${script_dir}/weak_scaling_openmp_rot"
done

for n in "${values[@]}"; do
  qsub \
    -N "ws_mpi_rot_${n}" \
    -v "N_PROCS=${n}" \
    -l "select=1:ncpus=${n}:mpiprocs=${n}:mem=2gb" \
    "${script_dir}/weak_scaling_mpi_rot"
done
