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
Integration example showing how to use privileged context self-distillation
with VERL's existing training infrastructure.

This demonstrates how to:
1. Extend the main_ppo training entry point
2. Switch between standard PPO/GRPO and privileged context distillation
3. Use command-line arguments to control training mode
"""

import os
from typing import Optional

from omegaconf import DictConfig, OmegaConf

import ray


def run_training_with_mode(config: DictConfig, training_mode: str = "ppo"):
    """
    Run training with specified mode.
    
    Args:
        config: Training configuration
        training_mode: One of ["ppo", "grpo", "privileged_context_distillation"]
    """
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Create trainer
    trainer = RayPPOTrainer(config=config)
    
    # Initialize workers
    trainer.init_workers()
    
    # Run training based on mode
    if training_mode == "privileged_context_distillation":
        print("=" * 80)
        print("Running Privileged Context Self-Distillation Training")
        print("=" * 80)
        trainer.fit_with_privileged_context_distillation()
    else:
        print("=" * 80)
        print(f"Running Standard {training_mode.upper()} Training")
        print("=" * 80)
        trainer.fit()
    
    # Cleanup
    ray.shutdown()


def create_config_from_args(args) -> DictConfig:
    """
    Create OmegaConf config from command-line arguments.
    
    This is a helper function that constructs a config similar to
    what you would get from a YAML file.
    """
    config = OmegaConf.create({
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
            "adv_estimator": args.algorithm,
        },
    })
    
    return config


def main_with_config_file(config_path: str, training_mode: str = "ppo"):
    """
    Run training using a YAML config file.
    
    Example:
        python integration_example.py --config my_config.yaml --mode privileged_context_distillation
    """
    # Load config from YAML
    config = OmegaConf.load(config_path)
    
    # Run training
    run_training_with_mode(config, training_mode)


def main_with_args():
    """
    Run training using command-line arguments.
    
    Example:
        python integration_example.py \
            --model_path deepseek-ai/deepseek-math-7b-base \
            --train_files ~/data/gsm8k/train.parquet \
            --mode privileged_context_distillation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="VERL Training with Multiple Modes")
    
    # Training mode
    parser.add_argument(
        "--mode",
        type=str,
        default="ppo",
        choices=["ppo", "grpo", "privileged_context_distillation"],
        help="Training mode (default: ppo)",
    )
    
    # Config file (alternative to individual args)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    
    # Model and data
    parser.add_argument("--model_path", type=str, help="Path to the base model")
    parser.add_argument("--train_files", type=str, help="Path to training data")
    parser.add_argument("--val_files", type=str, help="Path to validation data")
    
    # Training hyperparameters
    parser.add_argument("--total_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rollout_batch_size", type=int, default=64)
    parser.add_argument("--ppo_mini_batch_size", type=int, default=16)
    parser.add_argument("--ppo_micro_batch_size", type=int, default=2)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rollout_n", type=int, default=4)
    parser.add_argument("--rollout_tp", type=int, default=1)
    
    # Algorithm
    parser.add_argument("--algorithm", type=str, default="grpo", choices=["ppo", "grpo"])
    
    # Output and logging
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--project_name", type=str, default="verl_training")
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--test_freq", type=int, default=10)
    
    # Data processing
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_response_length", type=int, default=2048)
    
    args = parser.parse_args()
    
    # Use config file if provided, otherwise use args
    if args.config is not None:
        main_with_config_file(args.config, args.mode)
    else:
        if args.model_path is None or args.train_files is None:
            parser.error("--model_path and --train_files are required when not using --config")
        
        config = create_config_from_args(args)
        run_training_with_mode(config, args.mode)


# ============================================================================
# Example Usage Patterns
# ============================================================================

def example_1_standard_ppo():
    """Example 1: Standard PPO training (original VERL behavior)"""
    config = OmegaConf.create({
        "data": {"train_files": "~/data/gsm8k/train.parquet"},
        "actor_rollout_ref": {"model": {"path": "deepseek-ai/deepseek-math-7b-base"}},
        "trainer": {"total_training_steps": 100},
        "algorithm": {"adv_estimator": "ppo"},
    })
    
    run_training_with_mode(config, training_mode="ppo")


def example_2_privileged_context_distillation():
    """Example 2: Privileged context self-distillation (new feature)"""
    config = OmegaConf.create({
        "data": {
            "train_files": "~/data/gsm8k/train.parquet",  # Must include privileged_context
        },
        "actor_rollout_ref": {"model": {"path": "deepseek-ai/deepseek-math-7b-base"}},
        "trainer": {"total_training_steps": 100},
        "algorithm": {"adv_estimator": "grpo"},
    })
    
    run_training_with_mode(config, training_mode="privileged_context_distillation")


def example_3_from_yaml():
    """Example 3: Load config from YAML file"""
    # Create a sample YAML config
    config_yaml = """
data:
  train_files: ~/data/gsm8k/train.parquet
  val_files: ~/data/gsm8k/test.parquet
  train_batch_size: 16
  max_prompt_length: 2048

actor_rollout_ref:
  model:
    path: deepseek-ai/deepseek-math-7b-base
  actor:
    optim:
      lr: 1e-6
    ppo_mini_batch_size: 16
    ppo_epochs: 1
  rollout:
    name: sglang
    temperature: 1.0
    n: 4

trainer:
  total_training_steps: 100
  save_freq: 50
  test_freq: 10

algorithm:
  adv_estimator: grpo
"""
    
    # Save to file
    with open("/tmp/pcsd_config.yaml", "w") as f:
        f.write(config_yaml)
    
    # Load and run
    main_with_config_file("/tmp/pcsd_config.yaml", training_mode="privileged_context_distillation")


def example_4_conditional_training():
    """Example 4: Choose training mode based on data"""
    import os
    
    config = OmegaConf.create({
        "data": {"train_files": "~/data/gsm8k/train.parquet"},
        "actor_rollout_ref": {"model": {"path": "deepseek-ai/deepseek-math-7b-base"}},
        "trainer": {"total_training_steps": 100},
        "algorithm": {"adv_estimator": "grpo"},
    })
    
    # Check if data has privileged context (simplified check)
    # In practice, you'd check the actual data
    has_privileged_context = os.getenv("USE_PRIVILEGED_CONTEXT", "false").lower() == "true"
    
    if has_privileged_context:
        print("Data has privileged context, using distillation mode")
        run_training_with_mode(config, training_mode="privileged_context_distillation")
    else:
        print("Data doesn't have privileged context, using standard PPO mode")
        run_training_with_mode(config, training_mode="ppo")


# ============================================================================
# Integration with existing VERL main_ppo.py
# ============================================================================

def extended_main_ppo(config: DictConfig):
    """
    Extended version of verl.trainer.main_ppo that supports privileged context distillation.
    
    This can be used as a drop-in replacement for the standard main_ppo function.
    Simply add a "training_mode" field to your config:
    
    trainer:
      training_mode: privileged_context_distillation  # or "ppo", "grpo"
    """
    # Get training mode from config (default to standard PPO)
    training_mode = config.trainer.get("training_mode", "ppo")
    
    # Run training
    run_training_with_mode(config, training_mode)


if __name__ == "__main__":
    # Run main with command-line arguments
    main_with_args()

