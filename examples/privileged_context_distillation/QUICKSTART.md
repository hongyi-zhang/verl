# Quick Start: Privileged Context Self-Distillation

This guide helps you get started with privileged context self-distillation in VERL in 5 minutes.

## What is Privileged Context Self-Distillation?

A training method where a model learns to imitate itself when it has access to privileged information (like ground truth answers) that won't be available during inference.

**Example (GSM8K Math Problems):**
- **Student sees**: "What is 2 + 2?"
- **Teacher sees**: "What is 2 + 2?\n\nSolution: 2 + 2 = 4"
- **Goal**: Train the student to answer like the teacher, without seeing the solution during inference

## Installation

```bash
# Clone VERL
git clone https://github.com/volcengine/verl.git
cd verl

# Install dependencies (example for CUDA 12.6)
pip install -r requirements-cuda.txt
pip install -r requirements_sglang.txt
```

## Quick Start: 3 Steps

### Step 1: Prepare Your Data

Your dataset should have a `privileged_context` field in `extra_info`. For GSM8K, this is already done:

```python
# examples/data_preprocess/gsm8k.py (already modified)
python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

### Step 2: Configure Training

Edit `examples/privileged_context_distillation/train_gsm8k_pcsd.sh` or set environment variables:

```bash
export MODEL_PATH="deepseek-ai/deepseek-math-7b-base"
export DATA_PATH="~/data/gsm8k"
export TOTAL_STEPS=100
export ACTOR_NUM_GPUS=4
export ROLLOUT_NUM_GPUS=4
```

### Step 3: Run Training

```bash
bash examples/privileged_context_distillation/train_gsm8k_pcsd.sh
```

Or use the Python script:

```bash
python examples/privileged_context_distillation/train_pcsd.py \
    --model_path deepseek-ai/deepseek-math-7b-base \
    --data_path ~/data/gsm8k \
    --output_dir ./outputs/gsm8k_pcsd \
    --total_steps 100 \
    --batch_size 16 \
    --learning_rate 1e-6
```

## Custom Dataset

To use your own dataset with privileged context:

### 1. Add Privileged Context to Your Data

```python
# In your data preprocessing script
data = {
    "prompt": [
        {"role": "user", "content": "Your question here"}
    ],
    "extra_info": {
        "privileged_context": "Your privileged information here (e.g., answer, hints, reasoning)",
    }
}
```

### 2. Save as Parquet

```python
import datasets

dataset = datasets.Dataset.from_list(data_list)
dataset.to_parquet("train.parquet")
```

### 3. Run Training

```bash
python examples/privileged_context_distillation/train_pcsd.py \
    --model_path your/model/path \
    --data_path your/data/path \
    --output_dir ./outputs/custom
```

## Key Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--model_path` | Base model to train | Required | Any HF model |
| `--learning_rate` | Learning rate | 1e-6 | 1e-6 to 1e-5 |
| `--temperature` | Sampling temperature | 1.0 | 0.7 to 1.0 |
| `--rollout_n` | Responses per prompt | 4 | 4 to 16 |
| `--ppo_epochs` | Training epochs per batch | 1 | 1 to 3 |
| `--batch_size` | Training batch size | 16 | 8 to 32 |

## Troubleshooting

### Out of Memory (OOM)

**Solution 1**: Reduce batch size
```bash
python train_pcsd.py ... --batch_size 8 --ppo_micro_batch_size 1
```

**Solution 2**: Enable parameter offloading
```bash
# In your config
actor_rollout_ref:
  actor:
    fsdp_config:
      cpu_offload: true
```

### Slow Training

**Solution 1**: Increase rollout tensor parallel size
```bash
python train_pcsd.py ... --rollout_tp 2
```

**Solution 2**: Increase number of GPUs
```bash
python train_pcsd.py ... --actor_num_gpus 8 --rollout_num_gpus 8
```

### Model Not Learning

**Solution 1**: Increase learning rate
```bash
python train_pcsd.py ... --learning_rate 5e-6
```

**Solution 2**: Check privileged context quality
- Ensure `privileged_context` is informative
- Verify it's properly formatted and not truncated

**Solution 3**: Increase number of responses
```bash
python train_pcsd.py ... --rollout_n 8
```

## Monitoring Training

### Weights & Biases (W&B)

```bash
# Install wandb
pip install wandb
wandb login

# Update config
trainer:
  logger: ['wandb']
  project_name: "my_project"
  experiment_name: "gsm8k_pcsd"
```

### TensorBoard

```bash
# Update config
trainer:
  logger: ['tensorboard']

# Launch tensorboard
tensorboard --logdir ./outputs/gsm8k_pcsd/logs
```

### Console Logging

Training metrics are automatically printed:
- `distillation/loss`: Reverse KL loss
- `distillation/kl_div`: KL divergence between teacher and student
- `actor/lr`: Current learning rate
- `perf/mfu/actor`: Model FLOPs utilization

## Next Steps

- **Read the full documentation**: `examples/privileged_context_distillation/README.md`
- **Check implementation details**: `PRIVILEGED_CONTEXT_DISTILLATION_SUMMARY.md`
- **Experiment with hyperparameters**: Try different learning rates, temperatures, and batch sizes
- **Try your own dataset**: Add privileged context to your data

## Example: Training on GSM8K

```bash
# 1. Prepare data
python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

# 2. Set environment
export MODEL_PATH="deepseek-ai/deepseek-math-7b-base"
export DATA_PATH="~/data/gsm8k"
export TOTAL_STEPS=100

# 3. Train
bash examples/privileged_context_distillation/train_gsm8k_pcsd.sh

# 4. Monitor
# Check outputs/gsm8k_pcsd/logs/ for logs and checkpoints
```

Expected training time:
- 100 steps on 4 GPUs: ~30-60 minutes
- 1000 steps on 8 GPUs: ~4-8 hours

## Support

For questions or issues:
1. Check `examples/privileged_context_distillation/README.md`
2. Review `PRIVILEGED_CONTEXT_DISTILLATION_SUMMARY.md`
3. Open an issue on GitHub with error logs and configuration

## Citation

If you use this implementation, please cite:

```bibtex
@article{verl2024,
  title={VERL: A Unified Framework for Post-Training of Large Language Models},
  author={VERL Team},
  year={2024}
}
```

And the original distillation papers:

```bibtex
@article{hsieh2023distilling,
  title={Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes},
  author={Hsieh, Cheng-Yu and Li, Chun-Liang and Yeh, Chih-Kuan and Nakhost, Hootan and Fujii, Yasuhisa and Ratner, Alexander and Krishna, Ranjay and Lee, Chen-Yu and Pfister, Tomas},
  journal={arXiv preprint arXiv:2305.02301},
  year={2023}
}
```

