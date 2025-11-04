import asyncio
import concurrent.futures
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

from rllm.agents.agent import Action, BaseAgent, Trajectory
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.env_utils import (
    compute_mc_return,
    compute_trajectory_reward,
)
from rllm.misc import colorful_print
from rllm.parser import ChatTemplateParser

logger = logging.getLogger(__name__)


class AgentExecutionEngine:
    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=1,
        trajectory_timeout=None,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=64,
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}

        self.config = config
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.overlong_filter = overlong_filter
        self.log_snapshots = False
        if hasattr(self.config, "rllm"):
            try:
                self.log_snapshots = self.config.rllm.get("log_snapshots", False)
            except AttributeError:
                self.log_snapshots = getattr(self.config.rllm, "log_snapshots", False)

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]

        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e9)

        if env_class is not None:
            assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
        else:
            self.chat_parser = chat_parser

        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", {})  # for openai api requests
        # Optional tuning knobs (can be set via rllm.agent.engine_args)
        self.summary_max_tokens = kwargs.get("summary_max_tokens", None)
        self.action_max_tokens = kwargs.get("action_max_tokens", None)
        self.disable_response_length_limit = kwargs.get("disable_response_length_limit", False)
        # Per-call retry for summary generation; default high to be robust
        self.summary_retry_limit = int(kwargs.get("summary_retry_limit", 100))

        assert self.engine_name in ["openai", "verl"], "Currently only openai and verl are supported as rollout engine"
        if self.engine_name == "openai":
            from rllm.engine.rollout.openai_engine import OpenAIEngine

            self.rollout_engine = OpenAIEngine(
                **rollout_engine_args,
                api_retries=api_retries,
                tokenizer=self.tokenizer,
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                disable_thinking=kwargs.get("disable_thinking", False),
            )
        elif self.engine_name == "verl":
            from rllm.engine.rollout.verl_engine import VerlEngine

            self.rollout_engine = VerlEngine(
                config=self.config,
                rollout_manager=rollout_engine,
                tokenizer=self.tokenizer,
                disable_thinking=self.config.rllm.disable_thinking,
            )

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def get_model_response(self, prompt, application_id, **kwargs) -> str:
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response text

        Raises:
            NotImplementedError: If the engine type is not supported
        """

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        if self.engine_name == "openai":
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, enforce_max_prompt_length=False, **sampling_params)
            return output.text
        elif self.engine_name == "verl":
            meta_data = sampling_params.pop("meta_info", {})
            validate = meta_data.get("validate", False)
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, validate=validate, enforce_max_prompt_length=False, **sampling_params)
            return output.text
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def update_envs_and_agents(self, envs, agents):
        """
        Update the environments and agents.

        Args:
            envs: List of environments to use
            agents: List of agents to use
        """
        assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        self.envs = envs
        # For keeping track of the environment index in the batch.
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents
        self.n_parallel_agents = len(envs)

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agents[idx]
        env = self.envs[idx]
        # env_id = env.env_id
        
        # DEBUG: Add trajectory ID tracking to detect cross-contamination
        import logging
        trajectory_id = f"{application_id[:8]}_idx{idx}"
        agent._trajectory_id = trajectory_id  # Tag agent with this trajectory's ID
        logging.debug(f"[{trajectory_id}] Starting trajectory with agent={id(agent)}")

        termination_reason = None
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        # for step return
        episode_steps = []

        # Reset environment with the task using the executor
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        # Reset agent
        logging.debug(f"[{trajectory_id}] Resetting agent before trajectory start")
        agent.reset()
        # Update agent internal state from environment.
        agent.update_from_env(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )
        # Use agent-defined short prompt if available (e.g., summary + current obs),
        # otherwise fall back to full chat history.
        messages = agent.get_prompt() if hasattr(agent, "get_prompt") else agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=True, contains_generation_msg=True)
        prompt_token_len = len(prompt_tokens)
        # Note, this should never happen!
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}, retrying")

        for step_idx in range(self.max_steps):
            # Summary is generated after ACTION and applied in the same step (before next update_from_env)

            # Get action from agent using short prompt if provided
            prompt_messages = agent.get_prompt() if hasattr(agent, "get_prompt") else agent.chat_completions.copy()
            # Max tokens budget for the response
            if not self.enforce_max_prompt_length:
                if self.disable_response_length_limit:
                    # No episode-level limit; use per-call cap if provided
                    max_tokens = self.action_max_tokens or self.max_response_length
                else:
                    # Episode-level cumulative budget is based ONLY on data.max_response_length
                    remaining = self.max_response_length - response_token_len
                    # If no remaining budget, truncate the episode gracefully before calling the model
                    if remaining <= 0:
                        termination_reason = "TRUNCATION"
                        cur_step = agent.get_current_state()
                        if cur_step is not None:
                            cur_step.reward = 0.0
                            cur_step.done = True
                        break
                    # Per-call cap is the min of action_max_tokens and remaining episode budget
                    step_cap = self.action_max_tokens or remaining
                    max_tokens = min(step_cap, remaining)
            else:
                max_tokens = self.max_response_length

                # since max prompt is enforced, we filter out too long prompts.
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            # Server-side (vLLM) budget pre-check: ensure prompt fits into rollout prompt_length+response_length
            # and adjust per-call max_tokens accordingly to avoid negative max_tokens at the server.
            try:
                srv_prompt_len_cfg = getattr(self.config.actor_rollout_ref.rollout, "prompt_length", None)
                srv_resp_len_cfg = getattr(self.config.actor_rollout_ref.rollout, "response_length", None)
            except Exception:
                srv_prompt_len_cfg = None
                srv_resp_len_cfg = None
            if isinstance(srv_prompt_len_cfg, int) and srv_prompt_len_cfg > 0 and isinstance(srv_resp_len_cfg, int) and srv_resp_len_cfg > 0:
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len_tokens = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                overflow = max(0, prompt_len_tokens - srv_prompt_len_cfg)
                srv_remaining = srv_resp_len_cfg - overflow
                if srv_remaining <= 0:
                    # No capacity left at server for any new tokens; terminate before calling model
                    termination_reason = "TRUNCATION"
                    cur_step = agent.get_current_state()
                    if cur_step is not None:
                        cur_step.reward = 0.0
                        cur_step.done = True
                    break
                # Respect both local and server-side remaining budgets
                max_tokens = min(max_tokens, srv_remaining)

            # Pass integer max_tokens (non-negative ensured by pre-checks above)
            kwargs["max_tokens"] = int(max_tokens)

            start_time = time.time()
            # CRITICAL: Use unique request_id for each call to avoid conflicts with pending summary tasks
            import uuid
            action_request_id = f"{application_id}_action_step{step_idx}"
            logging.debug(f"[{trajectory_id}] Calling action generation with request_id={action_request_id}")
            response = await self.get_model_response(prompt_messages, action_request_id, **kwargs)
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time
            # Update steps
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
            }
            episode_steps.append(prompt_response_pair)

            # Update agent with model response
            action: Action = agent.update_from_model(response)
            action = action.action

            # Prepare concurrent tasks: world-model env step and (optional) summary generation
            action_lower = (action or "").lower()
            issued_stop = ("stop [" in action_lower) or ("send_msg_to_user" in action_lower)

            summary_task = None
            s_kwargs = None
            summary_prompt_messages = None
            if (not issued_stop) and hasattr(agent, "get_summary_prompt") and hasattr(agent, "update_summary_from_model"):
                try:
                    summary_prompt_messages = agent.get_summary_prompt()
                    if summary_prompt_messages:
                        s_kwargs = dict(kwargs)
                        s_kwargs["max_tokens"] = int(self.summary_max_tokens or min(256, self.max_response_length))
                        # Server-side (vLLM) pre-check: ensure summary prompt doesn't exceed server's prompt_length
                        try:
                            srv_prompt_len_cfg = getattr(self.config.actor_rollout_ref.rollout, "prompt_length", None)
                        except Exception:
                            srv_prompt_len_cfg = None
                        if isinstance(srv_prompt_len_cfg, int) and srv_prompt_len_cfg > 0:
                            summary_prompt_str = self.chat_parser.parse(summary_prompt_messages, add_generation_prompt=True, is_first_msg=True)
                            summary_prompt_len_tokens = len(self.tokenizer.encode(summary_prompt_str, add_special_tokens=False))
                            if summary_prompt_len_tokens > srv_prompt_len_cfg:
                                # Summary prompt exceeds vLLM server's single-call prompt limit
                                import logging
                                logging.warning(f"[{trajectory_id}] Summary prompt ({summary_prompt_len_tokens} tokens) exceeds server prompt_length ({srv_prompt_len_cfg}), skipping summary for step {step_idx}")
                                summary_prompt_messages = None
                        # launch summary task in background; apply at next step boundary
                        if summary_prompt_messages:
                            try:
                                import logging
                                # CRITICAL: Use unique request_id to avoid conflicts with action calls
                                summary_request_id = f"{application_id}_summary_step{step_idx}"
                                logging.debug(f"[{trajectory_id}] Launching summary generation with request_id={summary_request_id} for step {step_idx}, agent has {len(agent._trajectory.steps)} steps")
                                summary_task = asyncio.create_task(self.get_model_response(summary_prompt_messages, summary_request_id, **(s_kwargs or {})))
                            except Exception as e:
                                import logging
                                logging.error(f"[{trajectory_id}] Failed to create summary task: {e}")
                                summary_task = None
                except Exception:
                    summary_task = None

            if self.log_snapshots:
                self._log_snapshot(idx, step_idx, observation, action, prefix="action")

            # Environment step (summary and env run concurrently)
            try:
                env_start_time = time.time()
                # Use async version if available (e.g., WorldModelWebEnv) to avoid thread blocking
                if hasattr(env, 'step_async'):
                    env_future = asyncio.create_task(
                        asyncio.wait_for(env.step_async(action), timeout=max(0.001, self.trajectory_timeout - total_time))
                    )
                else:
                    # Fallback to executor for synchronous environments
                    env_future = asyncio.create_task(
                        asyncio.wait_for(loop.run_in_executor(self.executor, env.step, action), timeout=max(0.001, self.trajectory_timeout - total_time))
                    )
                next_observation, reward, done, info = await env_future
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n", "red")
                reward = 0
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Account timings
            env_delta_time = time.time() - env_start_time
            env_time += env_delta_time
            total_time += env_delta_time

            # CRITICAL: Apply summary BEFORE update_from_env to maintain correct message ordering
            # Messages order: [sys0, user0, asst0(action), summary0, sys1, user1, asst1(action), summary1, ...]
            if summary_task is not None:
                remaining_budget = self.trajectory_timeout - total_time
                if remaining_budget > 0:
                    _ws = time.time()
                    try:
                        import logging
                        logging.debug(f"[{trajectory_id}] Awaiting summary for step {step_idx}")
                        summary_response = await asyncio.wait_for(summary_task, timeout=max(0.001, remaining_budget))
                        total_time += time.time() - _ws
                        logging.debug(f"[{trajectory_id}] Applying summary for step {step_idx} BEFORE update_from_env")
                        agent.update_summary_from_model(summary_response)
                    except Exception as e:
                        import logging
                        logging.warning(f"[{trajectory_id}] Failed to await/apply summary: {e}")
                        total_time += time.time() - _ws
            
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len

            # Update agent internal state with next observation
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            # Check and convert to tokens if necessary
            assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assert env_messages is not None or mode != "Token", "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."

            # Include prior assistant message (e.g., summary) if present, so its tokens participate in training
            assistant_prev_msg_tokens, assistant_prev_msg_masks = [], []
            assistant_last_msg_tokens, assistant_last_msg_masks = [], []
            env_msg_tokens, env_msg_masks = [], []

            if mode == "Token":
                # find indices of assistant messages
                assistant_indices = [i for i, m in enumerate(chat_completions_messages) if m.get("role") == "assistant"]
                if len(assistant_indices) >= 2:
                    prev_assistant = chat_completions_messages[assistant_indices[-2]]
                    assistant_prev_msg_tokens, assistant_prev_msg_masks = convert_messages_to_tokens_and_masks([prev_assistant], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False)

            if assistant_message:
                assistant_last_msg_tokens, assistant_last_msg_masks = convert_messages_to_tokens_and_masks([assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False)
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True)

            # Update repsonse token length
            response_token_len += len(assistant_prev_msg_tokens) + len(assistant_last_msg_tokens) + len(env_msg_tokens)
            # Reached maximum number of tokens for the trajectory
            if (not self.enforce_max_prompt_length) and (not self.disable_response_length_limit) and (response_token_len >= self.max_response_length):
                # Truncation length
                truncation_length = self.max_response_length - response_token_len
                # Truncate the response and masks
                if truncation_length < 0:
                    truncated_response_tokens = (assistant_prev_msg_tokens + assistant_last_msg_tokens + env_msg_tokens)[:truncation_length]
                    truncated_response_masks = (assistant_prev_msg_masks + assistant_last_msg_masks + env_msg_masks)[:truncation_length]
                else:
                    # Edge case where the response is exactly the max response length.
                    truncated_response_tokens = assistant_prev_msg_tokens + assistant_last_msg_tokens + env_msg_tokens
                    truncated_response_masks = assistant_prev_msg_masks + assistant_last_msg_masks + env_msg_masks
                # Update token collections
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)

                cur_step = agent.get_current_state()
                # If the episode exceeds length after this step, zero reward only if the
                # model did not issue a STOP (i.e., no final answer to judge). If STOP was issued,
                # keep environment reward (already computed in env.step).
                has_stop_signal = False
                try:
                    has_stop_signal = bool(cur_step.info.get("has_stop_signal"))
                except Exception:
                    has_stop_signal = False
                if not has_stop_signal:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                # handle returning
                break

            # Update the token version of trajectory
            response_tokens.extend(assistant_prev_msg_tokens)
            response_masks.extend(assistant_prev_msg_masks)
            response_tokens.extend(assistant_last_msg_tokens)
            response_masks.extend(assistant_last_msg_masks)
            observation = next_observation
            # Per-step snapshot after env.step: print everything but reward (white color handled in _log_snapshot)
            if self.log_snapshots:
                self._log_snapshot(idx, step_idx, next_observation, None, done=done, info=info, prefix="env")

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Check if episode is done
            if done:
                # Before breaking, try to produce the final summary for the just-completed step

                # If we've reached the configured max steps (last loop index)
                # and the model did not issue STOP, prefer MAX_STEPS as the
                # termination reason over ENV_DONE.
                has_stop_signal = False
                try:
                    has_stop_signal = bool(cur_step.info.get("has_stop_signal"))
                except Exception:
                    has_stop_signal = False
                if step_idx == self.max_steps - 1 and not has_stop_signal:
                    termination_reason = "MAX_STEPS"
                else:
                    termination_reason = "ENV_DONE"
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        masked_out = False
        if self.overlong_filter:
            if termination_reason == "TRUNCATION" or termination_reason == "MAX_STEPS" or termination_reason == "TIMEOUT":
                # Mask out the entire response for overlong trajectories if the reward is 0.
                response_masks = [0] * len(response_masks)
                masked_out = True

        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward
        # Closing environment using the executor.
        await loop.run_in_executor(self.executor, env.close)
        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

        trajectory: Trajectory = agent.trajectory
        # Aggregate final trajectory statistics
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        if mode == "Text":
            return trajectory
        elif mode == "Token":
            token_result = {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "chat_completions": agent.chat_completions,
                "metrics": {
                    # Total number of steps taken in the trajectory
                    "steps": len(trajectory.steps),
                    # Time to calculate reward
                    "reward_time": reward_time,
                    # Total time spent in environment execution (env.step)
                    "env_time": env_time,
                    # Time to calculate response tokens
                    "llm_time": llm_time,
                    # Total time spent in the trajectory
                    "total_time": total_time,
                },
            }
            return token_result
        elif mode == "Conversation":
            return agent.chat_completions
        elif mode == "Step":
            steps_result = {
                "steps": episode_steps,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
            }
            return steps_result

    async def run_agent_trajectory_with_retry(self, idx, application_id, seed=0, mode="Text", **kwargs):
        for _ in range(self.retry_limit):
            try:
                return await asyncio.wait_for(self.run_agent_trajectory_async(idx, application_id=application_id, seed=seed, mode=mode, **kwargs), timeout=7200)
            except Exception:
                traceback.print_exc()
                continue
        traceback.print_exc()
        raise Exception(f"Trajectory {idx} cannot complete. Please check the log message")

    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
        if timing_raw is None:
            timing_raw = {}
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
        assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"  # type: ignore
        max_concurrency = self.n_parallel_agents
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

        if self.engine_name == "verl":
            self.rollout_engine.wake_up()

        async def launch_one_trajectory_task(env_idx: int):
            try:
                application_id = str(uuid.uuid4())
                result = await self.run_agent_trajectory_with_retry(
                    idx=env_idx,
                    application_id=application_id,
                    seed=reset_seed,
                    mode=mode,
                    **kwargs,
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise e
            return result

        # Create all N conceptual tasks. Their execution will be throttled by the semaphore
        # and the availability of agent/env indices.
        tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.envs))]

        tasks_completed = 0
        for coro in asyncio.as_completed(tasks_to_run):
            try:
                result = await coro
                tasks_completed += 1
                colorful_print(f"Number of Trajectories {tasks_completed}/{len(self.envs)} completed", "cyan")
                yield result
            except Exception as e:
                raise e

        if self.engine_name == "verl":
            self.rollout_engine.sleep()

        self.executor.shutdown(wait=False, cancel_futures=True)

    async def execute_tasks(self, tasks: list[dict]):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.

        Args:
            tasks: List of tasks to process
            max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)

        Returns:
            A list of trajectories, one for each task.
        """

        max_concurrent = self.n_parallel_agents

        # Initialize results list to store trajectories for all tasks
        all_trajectories = {}

        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        # Track completed trajectories
        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with semaphore:
                # Get an available index
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), "Agent is not initalized or not inheriting from BaseAgent"
                    self.agents[index].trajectory.task = task  # type: ignore
                    res = await self.run_agent_trajectory_async(index, application_id=task_id)
                    res.task = task
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                finally:
                    # Put the index back in the queue when done
                    await index_queue.put(index)

        # Run all tasks concurrently
        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
        return ordered_trajectories

    def _log_snapshot(self, traj_idx, step_idx, observation, action, reward=None, done=None, info=None, prefix=""):
        obs_str = ""
        if isinstance(observation, dict):
            obs_str = observation.get("axtree_txt") or observation.get("observation") or observation.get("text") or ""
        if not obs_str:
            obs_str = str(observation)
        obs_str = obs_str.replace("\n", " ").strip()
        if len(obs_str) > 50:
            obs_str = obs_str[:50] + "..."

        parts = [f"[traj {traj_idx} step {step_idx}"]
        if prefix:
            parts.append(prefix)
        if action is not None:
            action_str = str(action).replace("\n", " ").strip()
            if len(action_str) > 80:
                action_str = action_str[:80] + "..."
            parts.append(f"action={action_str}")
        if reward is not None:
            parts.append(f"reward={reward:.3f}")
        if done is not None:
            parts.append(f"done={done}")
        if info and isinstance(info, dict):
            reason = info.get("reward_reason") or info.get("termination_reason")
            if reason:
                parts.append(f"reason={reason}")
        parts.append(f"obs={obs_str}]")
        # Use white color for normal trajectory snapshots
        colorful_print(" ".join(parts), "white")

    def shutdown(self):
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown()
            self.executor = None


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
