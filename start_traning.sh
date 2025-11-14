set -e
cd /workspace/verl
export HF_HUB_ENABLE_HF_TRANSFER=0
export PCSD_DEBUG_STEP=10
CUDA_VISIBLE_DEVICES=0 python -m verl.trainer.main_ppo \
  trainer.project_name=pcsd_demo trainer.experiment_name=pcsd_news \
  trainer.nnodes=1 trainer.n_gpus_per_node=1 trainer.logger=console \
  trainer.total_epochs=1 trainer.val_before_train=false trainer.test_freq=-1 trainer.save_freq=100 \
  +trainer.debug_print_samples=true +trainer.debug_print_every=5 \
  trainer.total_epochs=10 \
  data.train_files=/workspace/pcsd_data/news_all.parquet \
  data.val_files=/workspace/pcsd_data/news_all.parquet \
  data.max_prompt_length=6666 data.max_response_length=1024 \
  data.truncation=right data.filter_overlong_prompts=true \
  data.train_batch_size=8 data.shuffle=false \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.rollout.response_length=1024 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.9 \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  critic.enable=false \
  +algorithm.privileged_distill=true