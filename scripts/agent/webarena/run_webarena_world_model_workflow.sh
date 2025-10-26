set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Locate the installed rllm package directory
RLLM_DIR=$(python3 -c "import rllm, os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

# Training data root (override by exporting WM_DATA_ROOT)
DATA_ROOT=${WM_DATA_ROOT:-${RLLM_DIR}/data/webarena}
if [ ! -d "${DATA_ROOT}" ]; then
    echo "Warning: expected dataset directory not found at ${DATA_ROOT}" >&2
fi

# World-model configuration can be overridden via environment variables before launch
export WM_MODEL_NAME=${WM_MODEL_NAME:-ms-w7gsz7wq}
export WM_BASE_URL=${WM_BASE_URL:-https://ms-w7gsz7wq-100041498772-sw.gw.ap-nanjing.ti.tencentcs.com/ms-w7gsz7wq/v1}
export WM_MAX_TOKENS=${WM_MAX_TOKENS:-8192}
export WM_TEMPERATURE=${WM_TEMPERATURE:-0.7}
export WM_TOP_P=${WM_TOP_P:-0.9}
export WM_ENV_MAX_STEPS=${WM_ENV_MAX_STEPS:-10}
export WM_AGENT_MAX_STEPS=${WM_AGENT_MAX_STEPS:-10}

python3 -m rllm.trainer.verl.train_agent_ppo \
    --config-name world_model_workflow \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_ROOT}/train.parquet \
    data.val_files=${DATA_ROOT}/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=64 \
    data.max_prompt_length=20480 \
    data.max_response_length=10240 \
    actor_rollout_ref.model.path=/data/dinghang/OpenRLHF/Llama-3.1-8B-Ins \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=False \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='7b-webarena_world_model_workflow' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=75 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=10 \
    "${@}"
