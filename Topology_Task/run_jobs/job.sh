#!/bin/bash
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --output=run_jobs/job_out_%j.log
#SBATCH --error=run_jobs/job_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G
#SBATCH --time=2-23:30:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch run_jobs/job.sh [preset] [main.py args...]
  sbatch run_jobs/job.sh --env-id bus14 --alg MAPPO --seed 0 --cuda true

  sbatch run_jobs/job.sh mappo14 \
  --track true \
  --wandb-entity corentin-plumet-epfl \
  --wandb-project marl2grid

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
  MARL2GRID_N_ENVS=40
  MARL2GRID_ROLLOUT_STEPS=20000
  MARL2GRID_N_STEPS=520
  MARL2GRID_EVAL_FREQ=20000
  MARL2GRID_TIME_LIMIT=2780
EOF
}

if [ -z "${PROJECT_DIR:-}" ]; then
  if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    if [ "$(basename "${SLURM_SUBMIT_DIR}")" = "run_jobs" ]; then
      PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
    else
      PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    fi
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  fi
fi
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-marl2grid}"
N_ENVS="${MARL2GRID_N_ENVS:-${SLURM_CPUS_PER_TASK:-40}}"
ROLLOUT_STEPS="${MARL2GRID_ROLLOUT_STEPS:-20000}"
if [ -n "${MARL2GRID_N_STEPS:-}" ]; then
  N_STEPS="${MARL2GRID_N_STEPS}"
else
  N_STEPS=$(( (ROLLOUT_STEPS + N_ENVS - 1) / N_ENVS ))
  N_STEPS=$(( ((N_STEPS + N_ENVS - 1) / N_ENVS) * N_ENVS ))
fi
if [ -n "${MARL2GRID_EVAL_FREQ:-}" ]; then
  EVAL_FREQ="${MARL2GRID_EVAL_FREQ}"
else
  EVAL_FREQ=$(( ((20000 + N_ENVS - 1) / N_ENVS) * N_ENVS ))
fi
TIME_LIMIT="${MARL2GRID_TIME_LIMIT:-2780}"

PRESET="${1:-mappo14}"
ARGS=(
  --cuda true
  --checkpoint true
  --n-threads 4
  --n-envs "${N_ENVS}"
  --n-steps "${N_STEPS}"
  --eval-freq "${EVAL_FREQ}"
  --time-limit "${TIME_LIMIT}"
)

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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_DIR}/run_jobs/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

echo "Project dir: ${PROJECT_DIR}"
echo "Started at: $(date)"
echo "Python: $(command -v python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "N_ENVS: ${N_ENVS}"
echo "N_STEPS: ${N_STEPS}"
echo "EVAL_FREQ: ${EVAL_FREQ}"
echo "Command: python main.py ${ARGS[*]}"

python main.py "${ARGS[@]}"

echo "Finished at: $(date)"
