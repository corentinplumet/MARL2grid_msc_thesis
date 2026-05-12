#!/bin/bash
# NUS Atlas SLURM launcher.
# Add an Atlas-specific account or partition here if your allocation requires one.
# Example:
# #SBATCH --partition=gpu
# #SBATCH --account=<your_atlas_project>
#SBATCH --job-name=marl2grid_atlas
#SBATCH --output=routput_jobs_atlas/job_out_%j.log
#SBATCH --error=routput_jobs_atlas/job_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=23:10:00
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: sbatch run_jobs/job_atlas.sh [preset] [main.py args...]

Presets: mappo14, mappo14_fast_noheuristic, qplex14, lagrmappo14
        table5_mappo14, table5_mappo14_fullobs, table5_qplex14, table5_lagrmappo14_l, table5_lagrmappo14_o
        table5_real_mappo14, table5_real_qplex14, table5_real_lagrmappo14_l, table5_real_lagrmappo14_o

Examples:
  Local smoke test:
    python main.py --cuda false --checkpoint false --n-threads 1 --n-envs 1 --n-steps 5 --eval-freq 1000000 --time-limit 2 --total-timesteps 20 --env-id bus14 --alg MAPPO --seed 0

  Cluster examples:
  sbatch run_jobs/job_atlas.sh
    python main.py --cuda true --checkpoint true --n-threads 1 --n-envs 40 --n-steps 520 --eval-freq 20000 --time-limit 110 --env-id bus14 --alg MAPPO

  sbatch run_jobs/job_atlas.sh qplex14 --seed 2 --track true
    python main.py --cuda true --checkpoint true --n-threads 1 --n-envs 40 --n-steps 520 --eval-freq 20000 --time-limit 110 --env-id bus14 --alg QPLEX --seed 2 --track true

  sbatch run_jobs/job_atlas.sh --env-id bus14 --alg MAPPO --seed 0
    python main.py --cuda true --checkpoint true --n-threads 1 --n-envs 40 --n-steps 520 --eval-freq 20000 --time-limit 110 --env-id bus14 --alg MAPPO --seed 0

  Heuristic bottleneck benchmark:
  sbatch run_jobs/job_atlas.sh mappo14_fast_noheuristic --seed 0
    python main.py --cuda true --checkpoint true --n-threads 1 --n-envs 40 --n-steps 120 --eval-freq 20000 --time-limit 120 --env-id bus14 --alg MAPPO --track false --use-heuristic false --seed 0

  Fast Table 5-style presets use paper hyperparameters with 40 parallel envs.
  Run one seed-0 job:
    sbatch run_jobs/job_atlas.sh table5_mappo14 --seed 0
    sbatch run_jobs/job_atlas.sh table5_mappo14_fullobs --seed 0
    sbatch run_jobs/job_atlas.sh table5_qplex14 --seed 0
    sbatch run_jobs/job_atlas.sh table5_lagrmappo14_l --seed 0
    sbatch run_jobs/job_atlas.sh table5_lagrmappo14_o --seed 0

  Real Table 5 presets use paper hyperparameters with 10 parallel envs and 2000 rollout steps.
  Run one seed-0 job:
    sbatch run_jobs/job_atlas.sh table5_real_mappo14 --seed 0
    sbatch run_jobs/job_atlas.sh table5_real_qplex14 --seed 0
    sbatch run_jobs/job_atlas.sh table5_real_lagrmappo14_l --seed 0
    sbatch run_jobs/job_atlas.sh table5_real_lagrmappo14_o --seed 0

Environment overrides: PROJECT_DIR, VENV_PATH, CONDA_ENV_NAME, N_ENVS, ROLLOUT_BATCH, N_STEPS, EVAL_FREQ, PY_TIME_LIMIT, CUDA, CHECKPOINT, TOTAL_TIMESTEPS, SEED, TRACK, DECENTRALIZED, WANDB_ENTITY, WANDB_PROJECT
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

case "${1:-mappo14}" in
  mappo14)
    [ "$#" -gt 0 ] && shift
    set_default ENV_ID bus14
    set_default ALG MAPPO
    ;;

  mappo14_fast_noheuristic)
    [ "$#" -gt 0 ] && shift
    set_default ENV_ID bus14
    set_default ALG MAPPO
    set_default USE_HEURISTIC false
    set_default TRACK false
    set_default N_ENVS 5
    set_default ROLLOUT_BATCH 4000
    set_default EVAL_FREQ 20000
    set_default PY_TIME_LIMIT 120
    set_default CUDA true
    set_default CHECKPOINT true
    ;;

  qplex14)
    [ "$#" -gt 0 ] && shift
    set_default ENV_ID bus14
    set_default ALG QPLEX
    ;;

  lagrmappo14)
    [ "$#" -gt 0 ] && shift
    set_default ENV_ID bus14
    set_default ALG LAGRMAPPO
    set_default CONSTRAINTS_TYPE 1
    ;;

  table5_mappo14)
    [ "$#" -gt 0 ] && shift
    set_table5_common MAPPO
    set_fast_table5_runtime
    set_table5_mappo
    ;;

  table5_mappo14_fullobs)
    [ "$#" -gt 0 ] && shift
    set_table5_common MAPPO
    set_fast_table5_runtime
    set_table5_mappo
    set_default DECENTRALIZED false
    ;;

  table5_qplex14)
    [ "$#" -gt 0 ] && shift
    set_table5_common QPLEX
    set_fast_table5_runtime
    set_table5_qplex
    ;;

  table5_lagrmappo14_l)
    [ "$#" -gt 0 ] && shift
    set_table5_common LAGRMAPPO
    set_fast_table5_runtime
    set_table5_lagrmappo 1 0
    ;;

  table5_lagrmappo14_o)
    [ "$#" -gt 0 ] && shift
    set_table5_common LAGRMAPPO
    set_fast_table5_runtime
    set_table5_lagrmappo 2 50
    ;;

  table5_real_mappo14)
    [ "$#" -gt 0 ] && shift
    set_table5_common MAPPO
    set_real_table5_runtime
    set_table5_mappo
    ;;

  table5_real_qplex14)
    [ "$#" -gt 0 ] && shift
    set_table5_common QPLEX
    set_real_table5_runtime
    set_table5_qplex
    ;;

  table5_real_lagrmappo14_l)
    [ "$#" -gt 0 ] && shift
    set_table5_common LAGRMAPPO
    set_real_table5_runtime
    set_table5_lagrmappo 1 0
    ;;

  table5_real_lagrmappo14_o)
    [ "$#" -gt 0 ] && shift
    set_table5_common LAGRMAPPO
    set_real_table5_runtime
    set_table5_lagrmappo 2 50
    ;;
esac

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
  elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED="${SLURM_ARRAY_TASK_ID}"
  fi
fi

get_slurm_time_limit() {
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v squeue >/dev/null 2>&1; then
    local limit
    limit="$(squeue -h -j "${SLURM_JOB_ID}" -o "%l" 2>/dev/null || true)"
    if [ -n "${limit}" ]; then
      echo "${limit}"
      return
    fi
  fi
  echo "${SLURM_TIMELIMIT:-unset}"
}

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
  echo "Requested GPU count: ${SLURM_GPUS:-unset}"
  echo "Allocated GPU IDs: ${SLURM_JOB_GPUS:-unset}"
  echo "Step GPU IDs: ${SLURM_STEP_GPUS:-unset}"
  echo "CUDA visible device IDs: ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "Time limit: $(get_slurm_time_limit)"
  echo "============================================"
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

mkdir -p "${PROJECT_DIR}/run_jobs"
cd "${PROJECT_DIR}"
source "${VENV_PATH}/bin/activate"

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
