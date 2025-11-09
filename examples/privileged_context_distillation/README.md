# Privileged Context Self-Distillation

This directory contains examples for training language models using **privileged context self-distillation**, an on-policy reinforcement learning algorithm where the model learns to imitate its own behavior when given privileged information (e.g., ground truth answers).

## Algorithm Overview

Privileged context self-distillation works as follows:

1. **Data Preparation**: Each training sample contains:
   - Student prompt: The original question (without privileged context)
   - Teacher prompt: The question + privileged context (e.g., ground truth answer)

2. **Training Loop** (per step):
   - Generate responses using the student prompt (without privileged context)
   - Compute teacher distributions: Forward pass with teacher prompt + generated responses
   - Compute student distributions: Forward pass with student prompt + generated responses
   - Update the model using reverse KL divergence: `KL(teacher || student)`

3. **Loss Function**: Reverse KL divergence encourages the student to cover all modes of the teacher distribution:
   ```
   L = KL(P_teacher || P_student) = Σ P_teacher(x) * (log P_teacher(x) - log P_student(x))
   ```

## Key Features

- **On-policy**: Responses are generated from the current policy at each training step
- **Self-distillation**: The same model serves as both teacher (with privileged context) and student (without privileged context)
- **Stop-gradient on teacher**: Teacher distributions are computed without gradients to prevent trivial solutions

## Usage

### 1. Prepare Dataset

The dataset should include a `privileged_context` field in the `extra_info` of each sample. For GSM8K:

```python
# In examples/data_preprocess/gsm8k.py
"extra_info": {
    "split": split,
    "index": idx,
    "answer": answer_raw,
    "question": question_raw,
    "privileged_context": answer_raw,  # Privileged context for distillation
}
```

### 2. Run Training

Use the provided training script:

```bash
bash examples/privileged_context_distillation/train_gsm8k_pcsd.sh
```

Or customize your training:

```bash
# Set environment variables
export MODEL_PATH="deepseek-ai/deepseek-math-7b-base"
export DATA_PATH="~/data/gsm8k"
export TOTAL_STEPS=100
export ACTOR_NUM_GPUS=4
export ROLLOUT_NUM_GPUS=4

# Run training
bash examples/privileged_context_distillation/train_gsm8k_pcsd.sh
```

### 3. Training Configuration

Key configuration options:

- `data.train_files`: Path to training data (parquet format)
- `data.val_files`: Path to validation data
- `actor_rollout_ref.model.path`: Path to the base model
- `actor_rollout_ref.actor.optim.lr`: Learning rate for actor updates
- `actor_rollout_ref.rollout.n`: Number of responses per prompt
- `trainer.total_training_steps`: Total training steps

## Code Structure

The implementation consists of:

1. **Dataset** (`verl/utils/dataset/rl_dataset.py`):
   - Modified `__getitem__` to construct teacher inputs with privileged context
   - Returns `teacher_input_ids`, `teacher_attention_mask`, `teacher_position_ids`

2. **Actor** (`verl/workers/actor/dp_actor.py`):
   - `compute_distributions()`: Computes logits/distributions over response tokens
   - `update_policy_with_distillation()`: Updates policy using reverse KL loss

3. **Worker** (`verl/workers/fsdp_workers.py`):
   - `compute_distributions()`: Wrapper for actor's compute_distributions
   - `update_actor_with_distillation()`: Wrapper for actor's update_policy_with_distillation

4. **Trainer** (`verl/trainer/ppo/ray_trainer.py`):
   - `fit_with_privileged_context_distillation()`: Main training loop for distillation

## Example: GSM8K

The GSM8K dataset includes step-by-step solutions as privileged context. During training:

- **Student** sees only the question: "What is 2 + 2?"
- **Teacher** sees question + solution: "What is 2 + 2?\n\nLet's solve step by step: 2 + 2 = 4\n#### 4"

The student learns to generate answers similar to what it would generate if it had seen the solution, without actually seeing it during rollout.

## Customization

### Using Different Privileged Context

Modify the dataset preprocessing to set `privileged_context` to any additional information:

```python
"extra_info": {
    "privileged_context": your_privileged_information,
}
```

### Adjusting Training Hyperparameters

Key hyperparameters to tune:

- **Temperature**: Controls sampling randomness (default: 1.0)
- **PPO Epochs**: Number of gradient steps per batch (default: 1)
- **Mini-batch Size**: Size of mini-batches for training (default: 16)
- **Learning Rate**: Learning rate for actor updates (default: 1e-6)

## References

This implementation is inspired by:

- [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301)
- [Specializing Smaller Language Models towards Multi-Step Reasoning](https://arxiv.org/abs/2301.12726)

## Notes

- The current implementation supports **FSDP + SGLang** for actor training and rollout
- Teacher distributions are computed with `torch.no_grad()` to save memory and prevent gradient flow
- The algorithm is on-policy: responses are regenerated at each training step

