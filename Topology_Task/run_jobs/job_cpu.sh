#!/bin/bash
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --output=run_jobs/job_out_%j.log
#SBATCH --error=run_jobs/job_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=academic
#SBATCH --gres=gpu:1

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch run_jobs/job.sh [preset] [main.py args...]
  sbatch run_jobs/job.sh --env-id bus14 --alg MAPPO --seed 0 --cuda true

Presets:
  mappo14        --env-id bus14 --alg MAPPO
  qplex14        --env-id bus14 --alg QPLEX
  lagrmappo14    --env-id bus14 --alg LAGRMAPPO --constraints-type 1

Examples:
  sbatch run_jobs/job.sh mappo14 --seed 2 --track true
  sbatch run_jobs/job.sh qplex14 --seed 0 --total-timesteps 25000000
  sbatch run_jobs/job.sh --env-id bus14 --alg MAPPO --use-heuristic false

Environment overrides:
  PROJECT_DIR=/path/to/Topology_Task
  VENV_PATH=/path/to/.venv
  CONDA_ENV_NAME=marl2grid
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-marl2grid}"

PRESET="${1:-mappo14}"
ARGS=(--cuda true --n-threads "${SLURM_CPUS_PER_TASK:-4}")

case "${PRESET}" in
  --help|-h)
    usage
    exit 0
    ;;
  mappo14)
    ARGS+=(--env-id bus14 --alg MAPPO)
    if [ "$#" -gt 0 ]; then shift; fi
    ;;
  qplex14)
    ARGS+=(--env-id bus14 --alg QPLEX)
    if [ "$#" -gt 0 ]; then shift; fi
    ;;
  lagrmappo14)
    ARGS+=(--env-id bus14 --alg LAGRMAPPO --constraints-type 1)
    if [ "$#" -gt 0 ]; then shift; fi
    ;;
esac

ARGS+=("$@")

mkdir -p "${PROJECT_DIR}/run_jobs"
cd "${PROJECT_DIR}"

if [ -d "${VENV_PATH}" ]; then
  source "${VENV_PATH}/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
else
  echo "No virtualenv found at ${VENV_PATH}, and conda is not available." >&2
  echo "Set VENV_PATH or CONDA_ENV_NAME before submitting the job." >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Project dir: ${PROJECT_DIR}"
echo "Started at: $(date)"
echo "Python: $(command -v python)"
echo "Command: python main.py ${ARGS[*]}"

python main.py "${ARGS[@]}"

echo "Finished at: $(date)"
