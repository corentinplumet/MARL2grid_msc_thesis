#!/bin/bash
# NUS Atlas PBS Pro launcher.
# Atlas uses PBS Pro/qsub rather than Slurm/sbatch.
# If your allocation requires a project, pass it at submission time:
#   qsub -P <your_nus_project_id> -- run_jobs/job_atlas.sh
# If Atlas reports "Unknown queue", list current queues with `hpc q` or `qstat -Q`
# and pass the live GPU queue at submission time with `qsub -q <queue> ...`.
##PBS -P <your_nus_project_id>
#PBS -N marl2grid_atlas
##PBS -q <gpu_queue>
#PBS -l select=1:ncpus=20:ngpus=1:mem=50gb
#PBS -l walltime=23:10:00
#PBS -j oe
#PBS -V

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  qsub -P <project_id> -- run_jobs/job_atlas.sh [preset] [main.py args...]
  qsub -P <project_id> -q <gpu_queue> -v PRESET=mappo14,SEED=0 run_jobs/job_atlas.sh
  qsub -P <project_id> -v PRESET=mappo14,SEED=0 run_jobs/job_atlas.sh

Presets: mappo14, mappo14_fast_noheuristic, qplex14, lagrmappo14
        table5_mappo14, table5_mappo14_fullobs, table5_qplex14, table5_lagrmappo14_l, table5_lagrmappo14_o
        table5_real_mappo14, table5_real_qplex14, table5_real_lagrmappo14_l, table5_real_lagrmappo14_o

Examples:
  Local smoke test:
    python main.py --cuda false --checkpoint false --n-threads 1 --n-envs 1 --n-steps 5 --eval-freq 1000000 --time-limit 2 --total-timesteps 20 --env-id bus14 --alg MAPPO --seed 0

  Cluster examples:
  qsub -P <project_id> -- run_jobs/job_atlas.sh
    python main.py --cuda true --checkpoint true --n-threads 1 --n-envs 20 --n-steps 520 --eval-freq 20000 --time-limit 1380 --env-id bus14 --alg MAPPO

  If your Atlas account requires an explicit queue:
  qsub -P <project_id> -q <gpu_queue> -v PRESET=mappo14,SEED=0 run_jobs/job_atlas.sh

  qsub -P <project_id> -- run_jobs/job_atlas.sh qplex14 --seed 2 --track true
    python main.py --cuda true --checkpoint true --n-threads 1 --n-envs 20 --n-steps 520 --eval-freq 20000 --time-limit 1380 --env-id bus14 --alg QPLEX --seed 2 --track true

  qsub -P <project_id> -v PRESET=mappo14,SEED=0 run_jobs/job_atlas.sh
    python main.py --cuda true --checkpoint true --n-threads 1 --n-envs 20 --n-steps 520 --eval-freq 20000 --time-limit 1380 --env-id bus14 --alg MAPPO --seed 0

  Heuristic bottleneck benchmark:
  qsub -P <project_id> -- run_jobs/job_atlas.sh mappo14_fast_noheuristic --seed 0
    python main.py --cuda true --checkpoint true --n-threads 1 --n-envs 5 --n-steps 120 --eval-freq 20000 --time-limit 120 --env-id bus14 --alg MAPPO --track false --use-heuristic false --seed 0

  Fast Table 5-style presets use paper hyperparameters with many parallel envs.
  Run one seed-0 job:
    qsub -P <project_id> -- run_jobs/job_atlas.sh table5_mappo14 --seed 0
    qsub -P <project_id> -- run_jobs/job_atlas.sh table5_mappo14_fullobs --seed 0
    qsub -P <project_id> -- run_jobs/job_atlas.sh table5_qplex14 --seed 0
    qsub -P <project_id> -- run_jobs/job_atlas.sh table5_lagrmappo14_l --seed 0
    qsub -P <project_id> -- run_jobs/job_atlas.sh table5_lagrmappo14_o --seed 0

  Real Table 5 presets use paper hyperparameters with 10 parallel envs and 2000 rollout steps.
  Run one seed-0 job:
    qsub -P <project_id> -- run_jobs/job_atlas.sh table5_real_mappo14 --seed 0
    qsub -P <project_id> -- run_jobs/job_atlas.sh table5_real_qplex14 --seed 0
    qsub -P <project_id> -- run_jobs/job_atlas.sh table5_real_lagrmappo14_l --seed 0
    qsub -P <project_id> -- run_jobs/job_atlas.sh table5_real_lagrmappo14_o --seed 0

Before submitting, check live queue names on Atlas with:
  hpc q
  qstat -Q

Environment overrides: PRESET, PROJECT_DIR, VENV_PATH, CONDA_ENV_NAME, ATLAS_EBENV_MODULE, LOAD_ATLAS_EBENV, DRY_RUN, N_ENVS, ROLLOUT_BATCH, N_STEPS, EVAL_FREQ, PY_TIME_LIMIT, CUDA, CHECKPOINT, TOTAL_TIMESTEPS, SEED, TRACK, DECENTRALIZED, WANDB_ENTITY, WANDB_PROJECT
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${PWD}}"
PROJECT_DIR="${PROJECT_DIR:-${SUBMIT_DIR}}"
if [ "$(basename "${PROJECT_DIR}")" = "run_jobs" ]; then
  PROJECT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
elif [ ! -f "${PROJECT_DIR}/main.py" ]; then
  PROJECT_DIR="${SCRIPT_PROJECT_DIR}"
fi

VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-marl2grid}"

set_default() {
  local name="$1"
  local value="$2"
  if [ -z "${!name:-}" ]; then
    printf -v "${name}" '%s' "${value}"
  fi
}

set_table5_common() {
  set_default ENV_ID bus14
  set_default ALG "$1"
  set_default USE_HEURISTIC false
  set_default TRACK true
  set_default TOTAL_TIMESTEPS 60000000
  set_default GAMMA 0.99
  set_default WANDB_ENTITY corentin-plumet-epfl
  set_default WANDB_PROJECT marl2grid
}

set_table5_mappo() {
  set_default MAX_GRAD_NORM 10
  set_default UPDATE_EPOCHS 80
  set_default N_MINIBATCHES 4
  set_default ACTOR_LR 3e-5
  set_default CRITIC_LR 3e-5
  set_default CLIP_COEF 0.2
}

set_table5_qplex() {
  set_default TRAIN_FREQ 100
  set_default TG_QNET_FREQ 2500
  set_default BUFFER_SIZE 1000000
  set_default BATCH_SIZE 128
  set_default LR 3e-5
  set_default EPS_DECAY_FRAC 0.5
}

set_table5_lagrmappo() {
  set_default CONSTRAINTS_TYPE "$1"
  set_default MAX_GRAD_NORM 10
  set_default UPDATE_EPOCHS 80
  set_default N_MINIBATCHES 4
  set_default ACTOR_LR 3e-5
  set_default CRITIC_LR 3e-5
  set_default CLIP_COEF 0.2
  set_default COST_THRESHOLD "$2"
  set_default LAG_MUL 0.0
  set_default LAG_LR 0.05
}

set_fast_table5_runtime() {
  set_default N_THREADS 1
  set_default N_ENVS 120 #40 #120
  set_default N_STEPS "$((N_ENVS * 2))"
  set_default ROLLOUT_BATCH "$((N_ENVS * N_STEPS))"
  set_default EVAL_FREQ "$((N_ENVS * N_STEPS * 5))"
  set_default PY_TIME_LIMIT 1380
  set_default CUDA true
  set_default CHECKPOINT true
}

set_real_table5_runtime() {
  set_default N_THREADS 1
  set_default N_ENVS 10
  set_default N_STEPS 2000
  set_default ROLLOUT_BATCH 20000
  set_default EVAL_FREQ 20000
  set_default PY_TIME_LIMIT 115
  set_default CUDA true
  set_default CHECKPOINT true
}

PRESET_NAME="${PRESET:-mappo14}"
if [ "$#" -gt 0 ]; then
  case "$1" in
    mappo14|mappo14_fast_noheuristic|qplex14|lagrmappo14|table5_mappo14|table5_mappo14_fullobs|table5_qplex14|table5_lagrmappo14_l|table5_lagrmappo14_o|table5_real_mappo14|table5_real_qplex14|table5_real_lagrmappo14_l|table5_real_lagrmappo14_o)
      PRESET_NAME="$1"
      shift
      ;;
    --*)
      ;;
    *)
      echo "Unknown preset '$1'. Run: bash run_jobs/job_atlas.sh --help" >&2
      exit 2
      ;;
  esac
fi

case "${PRESET_NAME}" in
  mappo14)
    set_default ENV_ID bus14
    set_default ALG MAPPO
    ;;

  mappo14_fast_noheuristic)
    set_default ENV_ID bus14
    set_default ALG MAPPO
    set_default USE_HEURISTIC false
    set_default TRACK false
    set_default N_ENVS 5
    set_default N_STEPS 120
    set_default ROLLOUT_BATCH 4000
    set_default EVAL_FREQ 20000
    set_default PY_TIME_LIMIT 120
    set_default CUDA true
    set_default CHECKPOINT true
    ;;

  qplex14)
    set_default ENV_ID bus14
    set_default ALG QPLEX
    ;;

  lagrmappo14)
    set_default ENV_ID bus14
    set_default ALG LAGRMAPPO
    set_default CONSTRAINTS_TYPE 1
    ;;

  table5_mappo14)
    set_table5_common MAPPO
    set_fast_table5_runtime
    set_table5_mappo
    ;;

  table5_mappo14_fullobs)
    set_table5_common MAPPO
    set_fast_table5_runtime
    set_table5_mappo
    set_default DECENTRALIZED false
    ;;

  table5_qplex14)
    set_table5_common QPLEX
    set_fast_table5_runtime
    set_table5_qplex
    ;;

  table5_lagrmappo14_l)
    set_table5_common LAGRMAPPO
    set_fast_table5_runtime
    set_table5_lagrmappo 1 0
    ;;

  table5_lagrmappo14_o)
    set_table5_common LAGRMAPPO
    set_fast_table5_runtime
    set_table5_lagrmappo 2 50
    ;;

  table5_real_mappo14)
    set_table5_common MAPPO
    set_real_table5_runtime
    set_table5_mappo
    ;;

  table5_real_qplex14)
    set_table5_common QPLEX
    set_real_table5_runtime
    set_table5_qplex
    ;;

  table5_real_lagrmappo14_l)
    set_table5_common LAGRMAPPO
    set_real_table5_runtime
    set_table5_lagrmappo 1 0
    ;;

  table5_real_lagrmappo14_o)
    set_table5_common LAGRMAPPO
    set_real_table5_runtime
    set_table5_lagrmappo 2 50
    ;;
esac

set_default N_THREADS 1
set_default N_ENVS 20
set_default N_STEPS 520
set_default ROLLOUT_BATCH "$((N_ENVS * N_STEPS))"
set_default EVAL_FREQ 20000
set_default PY_TIME_LIMIT 1380
set_default CUDA true
set_default CHECKPOINT true

has_cli_arg() {
  local flag="$1"
  shift
  for arg in "$@"; do
    case "${arg}" in
      "${flag}"|"${flag}="*) return 0 ;;
    esac
  done
  return 1
}

if ! has_cli_arg "--seed" "$@"; then
  if [ -n "${SEED:-}" ]; then
    :
  elif [ -n "${PBS_ARRAY_INDEX:-}" ]; then
    SEED="${PBS_ARRAY_INDEX}"
  elif [ -n "${PBS_ARRAYID:-}" ]; then
    SEED="${PBS_ARRAYID}"
  fi
fi

get_pbs_walltime() {
  if [ -n "${PBS_JOBID:-}" ] && command -v qstat >/dev/null 2>&1; then
    local limit
    limit="$(qstat -f "${PBS_JOBID}" 2>/dev/null | awk -F'= ' '/Resource_List.walltime/ {print $2; exit}' || true)"
    if [ -n "${limit}" ]; then
      echo "${limit}"
      return
    fi
  fi
  echo "unset"
}

print_resource_summary() {
  local nodefile_slots="unset"
  local node_list="unset"
  if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
    nodefile_slots="$(wc -l < "${PBS_NODEFILE}" | tr -d ' ')"
    node_list="$(sort -u "${PBS_NODEFILE}" | tr '\n' ' ')"
  fi

  echo "========== PBS resource summary =========="
  echo "Job id: ${PBS_JOBID:-unset}"
  echo "Job name: ${PBS_JOBNAME:-unset}"
  echo "Project: ${PBS_PROJECT:-unset}"
  echo "Queue: ${PBS_QUEUE:-unset}"
  echo "Submit dir: ${PBS_O_WORKDIR:-unset}"
  echo "Current host: $(hostname)"
  echo "Node file: ${PBS_NODEFILE:-unset}"
  echo "Node list: ${node_list}"
  echo "Nodefile slots: ${nodefile_slots}"
  echo "Requested ncpus: ${PBS_NCPUS:-unset}"
  echo "Requested ngpus: ${PBS_NGPUS:-unset}"
  echo "GPU file: ${PBS_GPUFILE:-unset}"
  echo "CUDA visible device IDs: ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "Walltime limit: $(get_pbs_walltime)"
  echo "========================================"
}

print_torch_cuda_summary() {
  python - <<'PY'
try:
    import torch

    print("========== PyTorch CUDA summary ==========")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count visible to job: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            print(f"CUDA device {idx}: {torch.cuda.get_device_name(idx)}")
    print("==========================================")
except Exception as exc:
    print("========== PyTorch CUDA summary ==========")
    print(f"Unable to query PyTorch CUDA state: {exc}")
    print("==========================================")
PY
}

add_arg() {
  local flag="$1"
  local value="${2:-}"
  if [ -n "${value}" ]; then
    ARGS+=("${flag}" "${value}")
  fi
}

ARGS=(
  --cuda "${CUDA}"
  --checkpoint "${CHECKPOINT}"
  --n-threads "${N_THREADS}"
  --n-envs "${N_ENVS}"
  --n-steps "${N_STEPS}"
  --eval-freq "${EVAL_FREQ}"
  --time-limit "${PY_TIME_LIMIT}"
)
add_arg --env-id "${ENV_ID}"
add_arg --alg "${ALG}"
add_arg --constraints-type "${CONSTRAINTS_TYPE:-}"
add_arg --use-heuristic "${USE_HEURISTIC:-}"
add_arg --decentralized "${DECENTRALIZED:-}"
add_arg --track "${TRACK:-}"
add_arg --wandb-entity "${WANDB_ENTITY:-}"
add_arg --wandb-project "${WANDB_PROJECT:-}"
add_arg --total-timesteps "${TOTAL_TIMESTEPS:-}"
add_arg --gamma "${GAMMA:-}"
add_arg --max-grad-norm "${MAX_GRAD_NORM:-}"
add_arg --update-epochs "${UPDATE_EPOCHS:-}"
add_arg --n-minibatches "${N_MINIBATCHES:-}"
add_arg --actor-lr "${ACTOR_LR:-}"
add_arg --critic-lr "${CRITIC_LR:-}"
add_arg --clip-coef "${CLIP_COEF:-}"
add_arg --train-freq "${TRAIN_FREQ:-}"
add_arg --tg-qnet-freq "${TG_QNET_FREQ:-}"
add_arg --buffer-size "${BUFFER_SIZE:-}"
add_arg --batch-size "${BATCH_SIZE:-}"
add_arg --lr "${LR:-}"
add_arg --eps-decay-frac "${EPS_DECAY_FRAC:-}"
add_arg --cost-threshold "${COST_THRESHOLD:-}"
add_arg --lag-mul "${LAG_MUL:-}"
add_arg --lag-lr "${LAG_LR:-}"
if ! has_cli_arg "--seed" "$@"; then
  add_arg --seed "${SEED:-}"
fi
ARGS+=("$@")

if [ "${DRY_RUN:-false}" = "true" ]; then
  echo "Project dir: ${PROJECT_DIR}"
  echo "Preset: ${PRESET_NAME}"
  echo "Command: python main.py ${ARGS[*]}"
  exit 0
fi

mkdir -p "${PROJECT_DIR}/run_jobs" "${PROJECT_DIR}/routput_jobs_atlas"
cd "${PROJECT_DIR}"
if [ -n "${PBS_JOBID:-}" ] && command -v tee >/dev/null 2>&1; then
  exec > >(tee -a "${PROJECT_DIR}/routput_jobs_atlas/job_${PBS_JOBID}.log") 2>&1
fi

if [ "${LOAD_ATLAS_EBENV:-true}" = "true" ] && [ -f /app1/ebenv ]; then
  if [ -n "${ATLAS_EBENV_MODULE:-}" ]; then
    source /app1/ebenv "${ATLAS_EBENV_MODULE}"
  else
    source /app1/ebenv
  fi
fi

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
print_torch_cuda_summary
echo "Vector envs: ${N_ENVS}, rollout steps: ${N_STEPS}, rollout batch: $((N_ENVS * N_STEPS))"
echo "Python time limit: ${PY_TIME_LIMIT} minutes"
echo "Command: python main.py ${ARGS[*]}"

python main.py "${ARGS[@]}"

echo "Finished at: $(date)"
