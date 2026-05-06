#!/bin/bash
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --output=run_jobs/job_out_%j.log
#SBATCH --error=run_jobs/job_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: sbatch run_jobs/job.sh [preset] [main.py args...]

Presets: mappo14, qplex14, lagrmappo14

Examples:
  sbatch run_jobs/job.sh
  sbatch run_jobs/job.sh qplex14 --seed 2 --track true
  sbatch run_jobs/job.sh --env-id bus14 --alg MAPPO --seed 0

Environment overrides: PROJECT_DIR, VENV_PATH, CONDA_ENV_NAME
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
if [ "$(basename "${PROJECT_DIR}")" = "run_jobs" ]; then
  PROJECT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
fi

VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-marl2grid}"

case "${1:-mappo14}" in
  mappo14)
    [ "$#" -gt 0 ] && shift
    set -- --env-id bus14 --alg MAPPO "$@"
    ;;
  qplex14)
    [ "$#" -gt 0 ] && shift
    set -- --env-id bus14 --alg QPLEX "$@"
    ;;
  lagrmappo14)
    [ "$#" -gt 0 ] && shift
    set -- --env-id bus14 --alg LAGRMAPPO --constraints-type 1 "$@"
    ;;
esac

N_ENVS="${N_ENVS:-${SLURM_CPUS_PER_TASK:-40}}"
ROLLOUT_BATCH="${ROLLOUT_BATCH:-20800}"
N_STEPS="${N_STEPS:-$(( ((ROLLOUT_BATCH + N_ENVS * N_ENVS - 1) / (N_ENVS * N_ENVS)) * N_ENVS ))}"
EVAL_FREQ="${EVAL_FREQ:-$((N_ENVS * 500))}"
PY_TIME_LIMIT="${PY_TIME_LIMIT:-110}"

ARGS=(
  --cuda true
  --checkpoint true
  --n-threads 1
  --n-envs "${N_ENVS}"
  --n-steps "${N_STEPS}"
  --eval-freq "${EVAL_FREQ}"
  --time-limit "${PY_TIME_LIMIT}"
  "$@"
)

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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_DIR}/run_jobs/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

echo "Project dir: ${PROJECT_DIR}"
echo "Started at: $(date)"
echo "Python: $(command -v python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "Vector envs: ${N_ENVS}, rollout steps: ${N_STEPS}, rollout batch: $((N_ENVS * N_STEPS))"
echo "Python time limit: ${PY_TIME_LIMIT} minutes"
echo "Command: python main.py ${ARGS[*]}"

python main.py "${ARGS[@]}"

echo "Finished at: $(date)"
