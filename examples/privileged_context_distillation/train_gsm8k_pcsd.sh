#!/bin/bash
# Example script for training with privileged context self-distillation on GSM8K
# This script demonstrates how to use the new training algorithm

set -x

# Model and data paths
MODEL_PATH=${MODEL_PATH:-"deepseek-ai/deepseek-math-7b-base"}
DATA_PATH=${DATA_PATH:-"~/data/gsm8k"}

# Training configuration
TOTAL_STEPS=${TOTAL_STEPS:-100}
BATCH_SIZE=${BATCH_SIZE:-16}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_EPOCHS=${PPO_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-6}

# Actor and rollout configuration
ACTOR_ROLLOUT_CONFIG=${ACTOR_ROLLOUT_CONFIG:-"fsdp"}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_N=${ROLLOUT_N:-4}  # Number of responses per prompt

# Resource allocation
ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-4}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-4}

# Temperature for generation and training
TEMPERATURE=${TEMPERATURE:-1.0}

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/gsm8k_pcsd"}
mkdir -p ${OUTPUT_DIR}

# Run training with privileged context self-distillation
python -m verl.trainer.main_ppo \
    data.train_files=${DATA_PATH}/train.parquet \
    data.val_files=${DATA_PATH}/test.parquet \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.logger=['console'] \
    trainer.project_name="privileged_context_distillation" \
    trainer.experiment_name="gsm8k_pcsd" \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=${OUTPUT_DIR} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    algorithm.adv_estimator=grpo \
    algorithm.training_mode=privileged_context_distillation

