


set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
STANDARD_ANSWER_K=${STANDARD_ANSWER_K:-1}
STANDARD_ANSWER_REWARD=${STANDARD_ANSWER_REWARD:-1.0}



# Resolve repository root that contains the packaged rllm module
RLLM_DIR=$(python3 -c "import rllm, os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

# World-model training data location can be overridden via WM_DATA_ROOT
DATA_ROOT=${WM_DATA_ROOT:-${RLLM_DIR}/data/webarena}
if [ ! -d "${DATA_ROOT}" ]; then
    echo "Warning: expected dataset directory not found at ${DATA_ROOT}" >&2
fi
export WANDB_API_KEY=""

# Ensure the learned world model implementation is available
WM_CLASS_PATH=${WM_CLASS_PATH:-rllm.world_model.web_world_model.WebWorldModel}
WM_MODEL_NAME=${WM_MODEL_NAME:-}
WM_API_KEY=${WM_API_KEY:-""}
WM_BASE_URL=${WM_BASE_URL:-}
WM_MAX_TOKENS=${WM_MAX_TOKENS:-8192}
WM_TEMPERATURE=${WM_TEMPERATURE:-0.7}
WM_TOP_P=${WM_TOP_P:-0.9}

# DAPO-based agent training (same args as GRPO variant, different entry/trainer)
python3 -m rllm.trainer.verl.train_agent_dapo \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=dapo \
    data.train_files=${DATA_ROOT}/webarena_trajs_train.parquet \
    data.val_files=${DATA_ROOT}/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=64 \
    data.max_prompt_length=32000 \
    data.max_response_length=16000 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.model.path=/data/public_models/Llama-3.1-8B-Instruct \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    +actor_rollout_ref.rollout.engine_kwargs.generation_config=vllm \
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
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=true \
    algorithm.filter_groups.metric=seq_final_reward \
    algorithm.filter_groups.max_num_gen_batches=2 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='gspo_7b-webarena_world_model' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=15 \
    trainer.default_hdfs_dir=null \
    rllm.env.name=world_model_web \
    +rllm.env.env_args.wm_class_path=${WM_CLASS_PATH} \
    +rllm.env.env_args.wm_model_name="${WM_MODEL_NAME}" \
    +rllm.env.env_args.vllm_urls="[${WM_BASE_URL}]" \
    +rllm.env.env_args.api_key="${WM_API_KEY}" \
    +rllm.env.env_args.use_chat_completions=true \
    +rllm.env.env_args.wm_max_new_tokens=${WM_MAX_TOKENS} \
    +rllm.env.env_args.temperature=${WM_TEMPERATURE} \
    +rllm.env.env_args.top_p=${WM_TOP_P} \
    +rllm.env.env_args.max_steps=5 \
    rllm.agent.name=world_model_web_agent \
    rllm.agent.max_steps=5 \
    rllm.standard_answer.inject_every_k=${STANDARD_ANSWER_K} \
    rllm.standard_answer.reward=${STANDARD_ANSWER_REWARD} \
    rllm.disable_thinking=True \
    +rllm.log_snapshots=true \
    trainer.total_epochs=10

#    agent.stepwise_advantage_mode="broadcast" \
#    actor_rollout_ref.rollout.enable_chunked_prefill=False \

