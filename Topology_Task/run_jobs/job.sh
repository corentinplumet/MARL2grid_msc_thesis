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
  Local smoke test:
    python main.py --cuda false --checkpoint false --n-threads 1 --n-envs 1 --n-steps 5 --eval-freq 1000000 --time-limit 2 --total-timesteps 20 --env-id bus14 --alg MAPPO --seed 0

  Cluster examples:
  sbatch run_jobs/job.sh
    python main.py --cuda true --checkpoint true --n-threads 1 --n-envs 40 --n-steps 520 --eval-freq 20000 --time-limit 110 --env-id bus14 --alg MAPPO

  sbatch run_jobs/job.sh qplex14 --seed 2 --track true
    python main.py --cuda true --checkpoint true --n-threads 1 --n-envs 40 --n-steps 520 --eval-freq 20000 --time-limit 110 --env-id bus14 --alg QPLEX --seed 2 --track true

  sbatch run_jobs/job.sh --env-id bus14 --alg MAPPO --seed 0
    python main.py --cuda true --checkpoint true --n-threads 1 --n-envs 40 --n-steps 520 --eval-freq 20000 --time-limit 110 --env-id bus14 --alg MAPPO --seed 0

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

print_resource_summary() {
  echo "========== SLURM resource summary =========="
  echo "Job id: ${SLURM_JOB_ID:-unset}"
  echo "Job name: ${SLURM_JOB_NAME:-unset}"
  echo "Account: ${SLURM_JOB_ACCOUNT:-unset}"
  echo "Partition: ${SLURM_JOB_PARTITION:-unset}"
  echo "QoS: ${SLURM_JOB_QOS:-unset}"
  echo "Submit dir: ${SLURM_SUBMIT_DIR:-unset}"
  echo "Node list: ${SLURM_JOB_NODELIST:-unset}"
  echo "Current host: $(hostname)"
  echo "Nodes: ${SLURM_NNODES:-unset}"
  echo "Tasks: ${SLURM_NTASKS:-unset}"
  echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unset}"
  echo "CPUs on node: ${SLURM_CPUS_ON_NODE:-unset}"
  echo "Memory per node: ${SLURM_MEM_PER_NODE:-unset} MB"
  echo "Memory per CPU: ${SLURM_MEM_PER_CPU:-unset} MB"
  echo "Requested GPUs: ${SLURM_GPUS:-unset}"
  echo "Allocated job GPUs: ${SLURM_JOB_GPUS:-unset}"
  echo "Step GPUs: ${SLURM_STEP_GPUS:-unset}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "Time limit: ${SLURM_TIMELIMIT:-unset}"

  if command -v scontrol >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "scontrol job details:"
    scontrol show job "${SLURM_JOB_ID}" || true
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU details:"
    nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,driver_version --format=csv || true
  fi

  if command -v free >/dev/null 2>&1; then
    echo "Node memory:"
    free -h || true
  fi

  if command -v lscpu >/dev/null 2>&1; then
    echo "CPU summary:"
    lscpu | grep -E '^(Architecture|CPU\\(s\\)|Thread\\(s\\) per core|Core\\(s\\) per socket|Socket\\(s\\)|Model name|NUMA node\\(s\\)):' || true
  fi
  echo "============================================"
}

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
print_resource_summary
echo "Python: $(command -v python)"
echo "Vector envs: ${N_ENVS}, rollout steps: ${N_STEPS}, rollout batch: $((N_ENVS * N_STEPS))"
echo "Python time limit: ${PY_TIME_LIMIT} minutes"
echo "Command: python main.py ${ARGS[*]}"

python main.py "${ARGS[@]}"

echo "Finished at: $(date)"
