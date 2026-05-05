#!/bin/bash
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --output=/home/plumet/msc_thesis/run_jobs/job_out_%j.log
#SBATCH --error=/home/plumet/msc_thesis/run_jobs/job_err_%j.log
#SBATCH --chdir=/home/plumet/msc_thesis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1



# Activate environment
source /home/plumet/msc_thesis/.venv/bin/activate

# === Base paths ===
BASE_DIR=/home/plumet/msc_thesis/src/RL
EXPERIMENT_CONFIG_DIR=$BASE_DIR/configs/experiments
EVAL_CONFIG_DIR=$BASE_DIR/configs/eval

# === Argument ===
MODE=$1
MODEL_PATH=$2

echo "Running mode: ${MODE}"
echo "Started at: $(date)"
if [ -n "$MODEL_PATH" ]; then
  echo "Model path: ${MODEL_PATH}"
fi

# === Dispatcher ===
case $MODE in

  baseline_random14)
    python $BASE_DIR/run_baseline.py \
      --config $EXPERIMENT_CONFIG_DIR/random_case14.toml
    ;;

  baseline_do_nothing14)
    python $BASE_DIR/run_baseline.py \
      --config $EXPERIMENT_CONFIG_DIR/do_nothing_case14.toml
    ;;

  baseline_do_nothing36)
    python $BASE_DIR/run_baseline.py \
      --config $EXPERIMENT_CONFIG_DIR/do_nothing_case36.toml
    ;;

  greedy14)
    python $BASE_DIR/run_baseline.py \
      --config $EXPERIMENT_CONFIG_DIR/greedy_case14.toml
    ;;

  greedy36)
    python $BASE_DIR/run_baseline.py \
      --config $EXPERIMENT_CONFIG_DIR/greedy_case36.toml
    ;;

  ppo14)
    python $BASE_DIR/train.py \
      --config $EXPERIMENT_CONFIG_DIR/ppo_vanilla_case14.toml \
      --device gpu
    ;;

  ppo14_rules)
    python $BASE_DIR/train.py \
      --config $EXPERIMENT_CONFIG_DIR/ppo_heuristic_rules_case14.toml \
      --device gpu
    ;;

  eval_ppo14)
    if [ -z "$MODEL_PATH" ]; then
      echo "Missing model path for eval_ppo14"
      echo "Usage: sbatch run_jobs/job.sh eval_ppo14 /abs/path/to/model.zip"
      exit 1
    fi
    python $BASE_DIR/evaluate_model.py \
      --config $EVAL_CONFIG_DIR/ppo_vanilla_case14.toml \
      --model-path $MODEL_PATH \
      --device gpu
    ;;

  eval_ppo14_rules)
    if [ -z "$MODEL_PATH" ]; then
      echo "Missing model path for eval_ppo14_rules"
      echo "Usage: sbatch run_jobs/job.sh eval_ppo14_rules /abs/path/to/model.zip"
      exit 1
    fi
    python $BASE_DIR/evaluate_model.py \
      --config $EVAL_CONFIG_DIR/ppo_heuristic_rules_case14.toml \
      --model-path $MODEL_PATH \
      --device gpu
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Available modes:"
    echo "  baseline_random14"
    echo "  baseline_do_nothing14"
    echo "  baseline_do_nothing36"
    echo "  greedy14"
    echo "  greedy36"
    echo "  ppo14"
    echo "  ppo14_rules"
    echo "  eval_ppo14 <model_path>"
    echo "  eval_ppo14_rules <model_path>"
    exit 1
    ;;

esac

echo "FINISHED AT: $(date)"
