import json
import random
import uuid
from dataclasses import dataclass
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.ray_trainer import (
    compute_data_metrics,
    compute_timing_metrics,
    RayWorkerGroup,
    marked_timer,
    reduce_metrics,
)

from rllm.parser import ChatTemplateParser

from .agent_ppo_trainer import AgentPPOTrainer


@dataclass
class StandardAnswerSample:
    prompt_tokens: torch.Tensor
    response_tokens: torch.Tensor
    response_mask: torch.Tensor


class AgentDAPOTrainer(AgentPPOTrainer):
    """Agent trainer variant that incorporates DAPO-style filtering while reusing
    the agent rollout pipeline from AgentPPOTrainer.

    This class avoids modifying `verl` by extending rllm's PPO agent wrapper and
    injecting DAPO-specific sampling logic (filter_groups) before optimization.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls=None,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
    ):
        if ray_worker_group_cls is None:
            ray_worker_group_cls = RayWorkerGroup
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            env_class=env_class,
            agent_class=agent_class,
            env_args=env_args,
            agent_args=agent_args,
        )
        self._init_standard_answer_support()

    def fit_agent(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        # validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                # DAPO-style accumulation to a prompt threshold
                batch_accum = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                fg_cfg = getattr(self.config.algorithm, "filter_groups", None)
                enable_fg = bool(getattr(fg_cfg, "enable", False)) if fg_cfg is not None else False
                fg_metric = getattr(fg_cfg, "metric", None) if fg_cfg is not None else None
                fg_max_gen = int(getattr(fg_cfg, "max_num_gen_batches", 0)) if fg_cfg is not None else 0

                desired_prompts = int(self.config.data.train_batch_size)

                with marked_timer("step", timing_raw):
                    while True:
                        # Build a fresh round and generate
                        round_batch: DataProto = DataProto.from_single_dict(batch_dict)
                        round_batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(round_batch.batch))], dtype=object
                        )
                        round_batch = round_batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                        )
                        # env rollout will generate
                        round_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

                        self.init_envs_and_agents(round_batch)

                        if self.config.rllm.stepwise_advantage.enable:
                            final_gen_batch_output = self.generate_agent_steps(
                                timing_raw=timing_raw, meta_info=round_batch.meta_info, uids=round_batch.non_tensor_batch["uid"]
                            )
                            repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                            round_batch = round_batch.sample_level_repeat(repeat_counts)
                            final_gen_batch_output.meta_info.pop("repeat_counts", None)
                            round_batch = round_batch.union(final_gen_batch_output)
                        else:
                            final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(
                                timing_raw=timing_raw, meta_info=round_batch.meta_info
                            )
                            round_batch = round_batch.union(final_gen_batch_output)
                            metrics.update(generate_metrics)

                        # optional value function is computed later on the accumulated batch

                        # reward model first (if enabled)
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(round_batch)
                            round_batch = round_batch.union(reward_tensor)

                        # rule/function rewards from trajectories
                        if "token_level_scores" not in round_batch.batch:
                            reward_tensor = self.reward_fn(round_batch)
                            round_batch.batch["token_level_scores"] = reward_tensor

                        # For DAPO we keep token_level_rewards equal to scores (no in-reward KL here)
                        round_batch.batch["token_level_rewards"] = round_batch.batch["token_level_scores"]

                        if self._should_use_standard_answers():
                            sa_stats = self._apply_standard_answers(round_batch)
                            metrics.update(sa_stats)
                        else:
                            metrics.setdefault("standard_answer/injected", 0)

                        # DAPO filter-groups (per-round) and accumulate successes
                        kept_this_round = 0
                        if enable_fg and fg_metric in {"seq_final_reward", "seq_reward"}:
                            if fg_metric == "seq_final_reward":
                                seq_vals = round_batch.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
                            else:
                                seq_vals = round_batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()

                            uids = round_batch.non_tensor_batch["uid"]
                            prompt_vals = {}
                            for uid, val in zip(uids, seq_vals, strict=True):
                                prompt_vals.setdefault(uid, []).append(val)
                            kept_uids = [uid for uid, vals in prompt_vals.items() if (len(vals) == 1 or np.std(vals) > 0)]
                            kept_mask = np.isin(uids, kept_uids)
                            if kept_mask.any():
                                round_batch = round_batch[kept_mask]
                                kept_this_round = len(kept_uids)
                            else:
                                kept_this_round = 0

                        # Accumulate
                        if (not enable_fg) or kept_this_round > 0:
                            batch_accum = round_batch if batch_accum is None else DataProto.concat([batch_accum, round_batch])
                            num_prompt_in_batch += kept_this_round if enable_fg else len(round_batch)

                        num_gen_batches += 1

                        # Stopping conditions
                        if num_prompt_in_batch >= desired_prompts:
                            break
                        if fg_max_gen > 0 and num_gen_batches >= fg_max_gen:
                            break

                # If nothing to train on, skip
                if batch_accum is None:
                    continue

                # Now compute logprob/entropy/ref/values/adv on accumulated batch
                with marked_timer("post_accum", timing_raw):
                    # recompute old_log_probs
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch_accum)
                    batch_accum = batch_accum.union(old_log_prob)

                    # entropy metric
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch_accum)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch_accum.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    metrics.update({"actor/entropy": entropy_agg.detach().item(), "train/num_gen_batches": num_gen_batches})
                    old_log_prob.batch.pop("entropys")
                    batch_accum = batch_accum.union(old_log_prob)

                    if self.use_reference_policy:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch_accum)
                        batch_accum = batch_accum.union(ref_log_prob)

                    # step-wise advantage handling
                    if self.config.rllm.stepwise_advantage.enable:
                        if self.config.rllm.stepwise_advantage.mode == "per_step":
                            batch_accum.batch["token_level_rewards"] = batch_accum.batch["mc_returns"]
                            batch_accum.non_tensor_batch["uid"] = batch_accum.non_tensor_batch["step_ids"]
                            is_pad_step = batch_accum.non_tensor_batch["is_pad_step"]
                            non_pad_step_indices = np.where(is_pad_step == False)[0]
                            batch_accum = batch_accum.select_idxs(non_pad_step_indices)
                        elif self.config.rllm.stepwise_advantage.mode == "broadcast":
                            is_last_step = batch_accum.non_tensor_batch["is_last_step"]
                            last_step_indices = np.where(is_last_step == True)[0]
                            other_step_indices = np.where(is_last_step == False)[0]
                            other_step_batch = batch_accum.select_idxs(other_step_indices)
                            batch_accum = batch_accum.select_idxs(last_step_indices)
                        else:
                            raise ValueError(f"Unsupported stepwise mode: {self.config.rllm.stepwise_advantage.mode}")

                    # values
                    if self.use_critic:
                        values = self.critic_wg.compute_values(batch_accum)
                        batch_accum = batch_accum.union(values)

                    # advantages
                    batch_accum = compute_advantage(
                        batch_accum,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                        config=self.config.algorithm,
                    )

                    if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                        self._stepwise_advantage_broadcast(batch_accum, other_step_batch=other_step_batch)
                        batch_accum = DataProto.concat([batch_accum, other_step_batch])

                if self.config.rllm.mask_truncated_samples:
                    mask = batch_accum.batch["attention_mask"][:, -1] == 1
                    batch_accum = batch_accum[~mask]

                batch_accum = self._pad_dataproto_to_world_size(batch=batch_accum)
                self._balance_batch(batch_accum, metrics=metrics)
                batch_accum.meta_info["global_token_num"] = torch.sum(batch_accum.batch["attention_mask"], dim=-1).tolist()

                if self.use_critic:
                    with marked_timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch_accum)
                    metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch_accum)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                    with marked_timer("testing", timing_raw):
                        val_metrics: dict = self._validate_agent()
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    with marked_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                metrics.update(compute_data_metrics(batch=batch_accum, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch_accum, timing_raw=timing_raw))
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _init_standard_answer_support(self):
        cfg = getattr(self.config.rllm, "standard_answer", {}) or {}
        self.standard_answer_interval = int(cfg.get("inject_every_k") or 0)
        self.standard_answer_reward = float(cfg.get("reward") or 1.0)
        seed = cfg.get("seed", None)
        if seed is None:
            seed = self.config.data.get("seed", 0)
        self.standard_answer_rng = random.Random(seed or 0)
        self.standard_answer_cache: dict[str, StandardAnswerSample] = {}
        self.standard_answer_prompt_limit = int(self.config.data.max_prompt_length)
        self.standard_answer_response_limit = int(self.config.data.max_response_length)
        self.standard_answer_parser = None
        if self.standard_answer_interval > 0:
            self.standard_answer_parser = ChatTemplateParser.get_parser(
                self.tokenizer, disable_thinking=self.config.rllm.disable_thinking
            )

    def _should_use_standard_answers(self) -> bool:
        return (
            self.standard_answer_interval > 0
            and self.standard_answer_parser is not None
            and self.global_steps % self.standard_answer_interval == 0
        )

    def _get_standard_answer_key(self, messages) -> str:
        try:
            return json.dumps(messages, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(messages)

    def _get_standard_answer_sample(self, messages) -> Optional[StandardAnswerSample]:
        if not isinstance(messages, (list, tuple)) or len(messages) == 0:
            return None
        key = self._get_standard_answer_key(messages)
        cached = self.standard_answer_cache.get(key)
        if cached is not None:
            return cached
        try:
            prompt_tokens, response_tokens, response_mask = self.standard_answer_parser.tokenize_and_mask_cumulative(messages)
        except ValueError:
            return None
        if prompt_tokens.numel() == 0 or response_tokens.numel() == 0:
            return None
        if prompt_tokens.numel() > self.standard_answer_prompt_limit or response_tokens.numel() > self.standard_answer_response_limit:
            return None
        sample = StandardAnswerSample(
            prompt_tokens=prompt_tokens.clone(),
            response_tokens=response_tokens.clone(),
            response_mask=response_mask.clone(),
        )
        self.standard_answer_cache[key] = sample
        return sample

    def _apply_standard_answers(self, batch: DataProto) -> dict[str, float]:
        stats = {
            "standard_answer/available_groups": 0,
            "standard_answer/filtered_groups": 0,
            "standard_answer/injected": 0,
        }
        standard_msgs = batch.non_tensor_batch.get("standard_answer_messages")
        uids = batch.non_tensor_batch.get("uid")
        if standard_msgs is None or uids is None:
            return stats

        # Group row indices by uid
        uid_to_indices: dict[str, list[int]] = {}
        for idx, uid in enumerate(uids):
            uid_to_indices.setdefault(str(uid), []).append(idx)

        for uid, indices in uid_to_indices.items():
            if not indices:
                continue
            example_msgs = standard_msgs[indices[0]] if len(standard_msgs) > indices[0] else None
            if example_msgs is None or (isinstance(example_msgs, float) and np.isnan(example_msgs)):
                continue
            stats["standard_answer/available_groups"] += 1
            sample = self._get_standard_answer_sample(example_msgs)
            if sample is None:
                stats["standard_answer/filtered_groups"] += 1
                continue
            replace_idx = self.standard_answer_rng.choice(indices)
            self._overwrite_batch_row(batch, replace_idx, sample)
            stats["standard_answer/injected"] += 1

        return stats

    def _overwrite_batch_row(self, batch: DataProto, row_idx: int, sample: StandardAnswerSample) -> None:
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        response_mask = batch.batch["response_mask"]
        input_ids = batch.batch["input_ids"]
        position_ids = batch.batch["position_ids"]
        token_scores = batch.batch["token_level_scores"]
        token_rewards = batch.batch.get("token_level_rewards")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        max_prompt_len = prompts.shape[1]
        max_response_len = responses.shape[1]
        device = prompts.device

        prompt_tokens = sample.prompt_tokens.to(device)
        response_tokens = sample.response_tokens.to(device)
        response_loss_mask = sample.response_mask.to(device)

        prompt_len = prompt_tokens.numel()
        response_len = response_tokens.numel()

        with torch.no_grad():
            prompt_row = torch.full((max_prompt_len,), pad_id, dtype=prompts.dtype, device=device)
            prompt_row[-prompt_len:] = prompt_tokens

            response_row = torch.full((max_response_len,), pad_id, dtype=responses.dtype, device=device)
            response_row[:response_len] = response_tokens

            prompts[row_idx] = prompt_row
            responses[row_idx] = response_row

            attn_prompt = torch.zeros((max_prompt_len,), dtype=attention_mask.dtype, device=device)
            attn_prompt[-prompt_len:] = 1
            attn_response = torch.zeros((max_response_len,), dtype=attention_mask.dtype, device=device)
            attn_response[:response_len] = 1
            attention_row = torch.cat([attn_prompt, attn_response], dim=0)
            attention_mask[row_idx] = attention_row

            response_mask_row = torch.zeros((max_response_len,), dtype=response_mask.dtype, device=device)
            response_mask_row[:response_len] = response_loss_mask
            response_mask[row_idx] = response_mask_row

            input_ids[row_idx] = torch.cat([prompt_row, response_row], dim=0)
            position_ids[row_idx] = (torch.cumsum(attention_row, dim=0) - 1) * attention_row

            token_scores_row = torch.zeros_like(token_scores[row_idx])
            if response_len > 0:
                token_scores_row[min(response_len - 1, token_scores_row.shape[0] - 1)] = self.standard_answer_reward
            token_scores[row_idx] = token_scores_row
            if token_rewards is not None:
                token_rewards[row_idx] = token_scores_row.clone()
