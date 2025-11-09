#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training script for privileged context self-distillation.

This script demonstrates how to train a language model using on-policy privileged context
self-distillation, where the model learns to imitate its own behavior when given privileged
information (e.g., ground truth answers or step-by-step solutions).

Usage:
    python train_pcsd.py \
        --model_path deepseek-ai/deepseek-math-7b-base \
        --data_path ~/data/gsm8k \
        --output_dir ./outputs/gsm8k_pcsd \
        --total_steps 100 \
        --batch_size 16
"""

import argparse
import os
import sys

import ray

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train with privileged context self-distillation")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--train_files", type=str, default=None, help="Path to training data (default: {data_path}/train.parquet)")
    parser.add_argument("--val_files", type=str, default=None, help="Path to validation data (default: {data_path}/test.parquet)")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Maximum prompt length")
    parser.add_argument("--max_response_length", type=int, default=2048, help="Maximum response length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/pcsd", help="Output directory")
    parser.add_argument("--total_steps", type=int, default=100, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--rollout_batch_size", type=int, default=64, help="Rollout batch size")
    parser.add_argument("--ppo_mini_batch_size", type=int, default=16, help="PPO mini-batch size")
    parser.add_argument("--ppo_micro_batch_size", type=int, default=2, help="PPO micro-batch size per GPU")
    parser.add_argument("--ppo_epochs", type=int, default=1, help="Number of PPO epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--rollout_n", type=int, default=4, help="Number of responses per prompt")
    
    # Resource arguments
    parser.add_argument("--actor_num_gpus", type=int, default=4, help="Number of GPUs for actor")
    parser.add_argument("--rollout_num_gpus", type=int, default=4, help="Number of GPUs for rollout")
    parser.add_argument("--rollout_tp", type=int, default=1, help="Tensor parallel size for rollout")
    
    # Checkpoint and logging arguments
    parser.add_argument("--save_freq", type=int, default=50, help="Checkpoint save frequency")
    parser.add_argument("--test_freq", type=int, default=10, help="Validation frequency")
    parser.add_argument("--project_name", type=str, default="privileged_context_distillation", help="Project name for logging")
    parser.add_argument("--experiment_name", type=str, default="pcsd", help="Experiment name")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set default paths if not provided
    if args.train_files is None:
        args.train_files = os.path.join(args.data_path, "train.parquet")
    if args.val_files is None:
        args.val_files = os.path.join(args.data_path, "test.parquet")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Ray
    ray.init()
    
    print("=" * 80)
    print("Privileged Context Self-Distillation Training")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Training data: {args.train_files}")
    print(f"Validation data: {args.val_files}")
    print(f"Output directory: {args.output_dir}")
    print(f"Total steps: {args.total_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print(f"Rollout N: {args.rollout_n}")
    print("=" * 80)
    
    # Build configuration
    from omegaconf import DictConfig, OmegaConf
    
    config = DictConfig({
        "data": {
            "train_files": args.train_files,
            "val_files": args.val_files,
            "train_batch_size": args.batch_size,
            "val_batch_size": args.batch_size,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
        },
        "actor_rollout_ref": {
            "model": {
                "path": args.model_path,
            },
            "actor": {
                "optim": {
                    "lr": args.learning_rate,
                },
                "ppo_mini_batch_size": args.ppo_mini_batch_size,
                "ppo_micro_batch_size_per_gpu": args.ppo_micro_batch_size,
                "ppo_epochs": args.ppo_epochs,
            },
            "rollout": {
                "name": "sglang",
                "temperature": args.temperature,
                "n": args.rollout_n,
                "rollout_batch_size": args.rollout_batch_size,
                "tensor_model_parallel_size": args.rollout_tp,
            },
        },
        "trainer": {
            "total_training_steps": args.total_steps,
            "logger": ["console"],
            "project_name": args.project_name,
            "experiment_name": args.experiment_name,
            "save_freq": args.save_freq,
            "test_freq": args.test_freq,
            "default_hdfs_dir": args.output_dir,
            "default_local_dir": args.output_dir,
        },
        "algorithm": {
            "adv_estimator": "grpo",
        },
    })
    
    # Create trainer
    trainer = RayPPOTrainer(config=config)
    
    # Initialize workers
    trainer.init_workers()
    
    # Run training with privileged context distillation
    print("\nStarting privileged context self-distillation training...")
    trainer.fit_with_privileged_context_distillation()
    
    print("\nTraining completed successfully!")
    print(f"Checkpoints saved to: {args.output_dir}")
    
    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    main()

