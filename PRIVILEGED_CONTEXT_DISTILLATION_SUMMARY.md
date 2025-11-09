# Privileged Context Self-Distillation Implementation Summary

This document summarizes the implementation of on-policy privileged context self-distillation for VERL.

## Overview

Privileged context self-distillation is a training algorithm where a language model learns to imitate its own behavior when given privileged information (e.g., ground truth answers, step-by-step solutions) that is not available during inference. The model serves as both teacher (with privileged context) and student (without privileged context).

## Implementation Details

### 1. Dataset Modifications (`verl/utils/dataset/rl_dataset.py`)

**Changes:**
- Added support for `privileged_context` field in `extra_info`
- Modified `__getitem__` to construct teacher inputs with privileged context appended to the last message
- Returns additional tensors: `teacher_input_ids`, `teacher_attention_mask`, `teacher_position_ids`

**Key Features:**
- Automatically detects if `privileged_context` is present in the data
- Handles both text-only and multimodal content
- Supports various vision position encoding schemes (Qwen2VL, Glm4v)

### 2. Actor Modifications (`verl/workers/actor/dp_actor.py`)

**New Methods:**

#### `_forward_micro_batch_distributions(micro_batch, temperature)`
- Performs forward pass and returns logits (distributions) over response tokens
- Returns: `(batch_size, response_length, vocab_size)` tensor
- Uses `torch.autocast` for mixed precision training

#### `compute_distributions(data: DataProto)`
- Wrapper that batches the forward pass for distribution computation
- Supports dynamic batch sizing
- Returns logits without computing log probabilities

#### `update_policy_with_distillation(data: DataProto)`
- Main training method for privileged context self-distillation
- Workflow:
  1. Compute teacher distributions (with privileged context) - **no gradient**
  2. Compute student distributions (without privileged context) - **with gradient**
  3. Calculate reverse KL divergence loss
  4. Backpropagate through student path only
- Returns metrics including loss and KL divergence

**Key Implementation Details:**
- Teacher distributions computed with `torch.no_grad()` to save memory and prevent gradient flow
- Reverse KL loss: `KL(teacher || student) = Σ P_teacher(x) * (log P_teacher(x) - log P_student(x))`
- Masks invalid tokens using `response_mask`
- Supports multiple PPO epochs and mini-batch training

### 3. Worker Modifications (`verl/workers/fsdp_workers.py`)

**New Methods:**

#### `compute_distributions(data: DataProto)`
- RPC wrapper for `actor.compute_distributions()`
- Handles FSDP model offloading/onloading
- Manages Ulysses sharding
- Returns `DataProto` with logits

#### `update_actor_with_distillation(data: DataProto)`
- RPC wrapper for `actor.update_policy_with_distillation()`
- Handles FSDP model and optimizer offloading/onloading
- Computes MFU (model FLOPs utilization)
- Updates learning rate scheduler
- Returns metrics including training statistics

### 4. Trainer Modifications (`verl/trainer/ppo/ray_trainer.py`)

**New Method:**

#### `fit_with_privileged_context_distillation()`
- Main training loop for privileged context self-distillation
- Workflow per training step:
  1. Load batch with teacher inputs
  2. Generate rollout sequences (student rollout without privileged context)
  3. Combine responses with both student and teacher prompts
  4. Update actor using distillation loss
  5. Log metrics, save checkpoints, and validate

**Key Features:**
- Supports async rollout mode
- Includes profiling and timing metrics
- Handles checkpoint saving with ESI expiration checks
- Compatible with validation and testing workflows

### 5. Dataset Preprocessing (`examples/data_preprocess/gsm8k.py`)

**Changes:**
- Added `privileged_context` field to `extra_info`
- Uses `answer_raw` (full step-by-step solution) as privileged context

## Usage

### Step 1: Prepare Dataset

Ensure your dataset includes the `privileged_context` field:

```python
"extra_info": {
    "privileged_context": your_privileged_information,
}
```

For GSM8K, this is automatically set to the full solution.

### Step 2: Run Training

Using the shell script:
```bash
bash examples/privileged_context_distillation/train_gsm8k_pcsd.sh
```

Or using the Python script:
```bash
python examples/privileged_context_distillation/train_pcsd.py \
    --model_path deepseek-ai/deepseek-math-7b-base \
    --data_path ~/data/gsm8k \
    --output_dir ./outputs/gsm8k_pcsd \
    --total_steps 100
```

### Step 3: Call the Training Method

In your training code:
```python
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

trainer = RayPPOTrainer(config=config)
trainer.init_workers()
trainer.fit_with_privileged_context_distillation()
```

## Algorithm Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load Batch                                               │
│    - Student prompt: "What is 2+2?"                         │
│    - Teacher prompt: "What is 2+2?\n[Solution: ...]"       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Generate Responses (Student Rollout)                     │
│    - Uses student prompt (without privileged context)       │
│    - Generates N responses per prompt                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Compute Teacher Distributions (NO GRADIENT)              │
│    - Input: Teacher prompt + generated responses            │
│    - Output: P_teacher(token | teacher_context, response)   │
│    - Stop gradient: .detach() or torch.no_grad()           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Compute Student Distributions (WITH GRADIENT)            │
│    - Input: Student prompt + generated responses            │
│    - Output: P_student(token | student_context, response)   │
│    - Keep gradient for backpropagation                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Compute Reverse KL Loss                                  │
│    - Loss = Σ P_teacher * (log P_teacher - log P_student)  │
│    - Masked by response_mask to ignore padding             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Backpropagate and Update                                 │
│    - Gradient flows through student path only               │
│    - Update model parameters                                │
│    - Step learning rate scheduler                           │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Key Hyperparameters

- **Temperature** (`rollout.temperature`): Controls sampling randomness (default: 1.0)
- **PPO Epochs** (`actor.ppo_epochs`): Number of gradient steps per batch (default: 1)
- **Mini-batch Size** (`actor.ppo_mini_batch_size`): Size of mini-batches for training
- **Learning Rate** (`actor.optim.lr`): Learning rate for actor updates (default: 1e-6)
- **Rollout N** (`rollout.n`): Number of responses per prompt (default: 4)

### Example Configuration

```yaml
data:
  train_files: ~/data/gsm8k/train.parquet
  val_files: ~/data/gsm8k/test.parquet
  max_prompt_length: 2048
  max_response_length: 2048

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
```

## Supported Backends

Currently supports:
- **Actor Training**: FSDP (Fully Sharded Data Parallel)
- **Rollout**: SGLang (for efficient text generation)

Future support planned for:
- Megatron-Core actor training
- vLLM rollout backend

## Memory Optimization

The implementation includes several memory optimizations:

1. **Stop-gradient on Teacher**: Teacher distributions computed with `torch.no_grad()`
2. **Mixed Precision**: Uses `torch.autocast` with bfloat16
3. **Parameter Offloading**: Supports FSDP model and optimizer offloading
4. **Dynamic Batching**: Supports dynamic batch sizing based on token count

## Testing

Test files created:
- `examples/privileged_context_distillation/train_gsm8k_pcsd.sh` - Shell script for GSM8K
- `examples/privileged_context_distillation/train_pcsd.py` - Python training script
- `examples/privileged_context_distillation/README.md` - Documentation

## Files Modified

1. `verl/utils/dataset/rl_dataset.py` - Dataset with teacher inputs
2. `verl/workers/actor/dp_actor.py` - Actor with distribution computation and distillation loss
3. `verl/workers/fsdp_workers.py` - Worker wrappers for new methods
4. `verl/trainer/ppo/ray_trainer.py` - Trainer with distillation training loop
5. `examples/data_preprocess/gsm8k.py` - GSM8K preprocessing with privileged context

## References

- [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301)
- [Specializing Smaller Language Models towards Multi-Step Reasoning](https://arxiv.org/abs/2301.12726)

## Future Work

Potential improvements:
1. Support for Megatron-Core actor training
2. Support for vLLM rollout backend
3. Mixed distillation + RL training (combine privileged context distillation with reward signals)
4. Curriculum learning with gradually reducing privileged context
5. Multi-teacher distillation (using multiple sources of privileged information)

