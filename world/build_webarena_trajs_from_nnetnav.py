#!/usr/bin/env python3
"""
将 stanfordnlp/nnetnav-* jsonl 原始样本解析成 WebArena 训练数据，
并为每条样本生成带“标准答案”轨迹（末尾不足 10 步的窗口）。

⚠️ 说明：
- 本脚本只做字符串解析与重组，不会调用任何额外 LLM / 补全模型。
- 对每个窗口样本同时输出 1 条与现有训练 schema 兼容的 base sample，
  以及完整的标准轨迹字段，方便 RL 训练与数据校验。
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:  # pragma: no cover - optional dependency
    torch = importlib.import_module("torch")
else:  # pragma: no cover - optional dependency
    torch = None

transformers_spec = importlib.util.find_spec("transformers")
if transformers_spec is not None:  # pragma: no cover - optional dependency
    transformers = importlib.import_module("transformers")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer
else:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None

pd = None
pd_spec = importlib.util.find_spec("pandas")
if pd_spec is not None:
    pd = importlib.import_module("pandas")

# Optional dependency: requests for vLLM OpenAI-compatible server
requests = None
requests_spec = importlib.util.find_spec("requests")
if requests_spec is not None:
    requests = importlib.import_module("requests")

SYSTEM_PROMPT = """You are an AI assistant performing tasks on a web browser. You will be provided with task objective, current step, web page observations, interaction history and previous taked notes. You need to issue an action for this step.

Generate the response in the following format:
INTERACTION HISTORY SUMMARY:
Emphasize all important details in the INTERACTION HISTORY section.

OBSERVATION DESCRIPTION:
Describe information in the CURRENT OBSERVATION section. Emphasize elements and features that are relevant or potentially helpful for fulfilling the objective in detail.

REASON:
Provide your rationale for proposing the subsequent action commands here.

ACTION:
Select your action here.

OBSERVATION HIGHLIGHT:
List the numerical ids of elements on the current webpage based on which you would issue your action. Also include elements on the current webpage you would attend to if you fail in the future and have to restore to this step. Don't include elements from the previous pages. Select elements at a higher hierarchical level if most their children nodes are considered crucial. Sort by relevance and potential values from high to low, and separate the ids with commas. E.g., `1321, 52, 756, 838`.

You are ONLY allowed to use the following action commands. Strictly adheres to the given format. Only issue one single action.
Use the following actions:
- click [id]: To click on an element with its numerical ID on the webpage. E.g., `click [7]` If clicking on a specific element doesn't trigger the transition to your desired web state, this is due to the element's lack of interactivity or GUI visibility. In such cases, move on to interact with OTHER similar or relevant elements INSTEAD.
- type [id] [content] [press_enter_after=0|1]: To type content into a field with a specific ID. By default, the "Enter" key is pressed after typing unless `press_enter_after` is set to 0. E.g., `type [15] [Carnegie Mellon University] [1]` If you can't find what you're looking for on your first attempt, consider refining your search keywords by breaking them down or trying related terms.
- stop [answer]: To stop interaction and return response. Present your answer within the brackets. If and only if the task doesn't require a textual answer or appears insurmountable, indicate "N/A" and additional reasons and all relevant information you gather as the answer. Otherwise, including "N/A" will be penalized. E.g., `stop [5h 47min]`
- go_back: To return to the previously viewed page."""

ACTION_PATTERN = re.compile(r"```(.*?)```", re.DOTALL)
STOP_PATTERN = re.compile(r"(stop\s*\[[^\]]+\])", re.IGNORECASE)
CODE_FENCE = re.compile(r"^```|```$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        type=Path,
        required=True,
        help="stanfordnlp/nnetnav-* 格式的 jsonl 文件",
    )
    parser.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("webarena_trajs.jsonl"),
        help="输出 jsonl 路径",
    )
    parser.add_argument(
        "--output_parquet",
        type=Path,
        default=Path("webarena_trajs.parquet"),
        help="输出 parquet 路径",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--completion_model",
        type=str,
        default=None,
        help="可选：用于补全标准答案的 HuggingFace Causal LM（本地路径或模型名）",
    )
    parser.add_argument(
        "--completion_device",
        type=str,
        default="cpu",
        help="加载 completion model 的设备，如 cpu / cuda:0",
    )
    parser.add_argument(
        "--completion_max_new_tokens",
        type=int,
        default=512,
        help="补全文本最大生成长度",
    )
    parser.add_argument(
        "--completion_temperature",
        type=float,
        default=0.6,
        help="补全采样温度，0 表示贪心",
    )
    parser.add_argument(
        "--completion_trust_remote_code",
        action="store_true",
        help="加载 completion model 时允许 trust_remote_code=True",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="transformers device_map, e.g. 'auto' to shard across visible GPUs; use 'none' to disable",
    )
    # vLLM OpenAI-compatible server（可选）
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="",
        help="vLLM OpenAI 兼容服务 base URL（如 http://127.0.0.1:8000/v1 或 http://host:port）。若提供则优先使用 vLLM 生成 assistant。",
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default="",
        help="vLLM OpenAI 兼容服务 API key（如启用鉴权时）。",
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default="",
        help="vLLM 服务端提供/注册的模型名（对应 --served-model-name）。",
    )
    parser.add_argument(
        "--vllm_request_timeout",
        type=float,
        default=120.0,
        help="vLLM 请求超时时间（秒）。",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="并行处理任务组的线程数（建议从 4 开始）。",
    )
    parser.add_argument(
        "--train_jsonl",
        type=Path,
        default=Path("webarena_trajs.train.jsonl"),
        help="训练集 jsonl 输出路径",
    )
    parser.add_argument(
        "--test_jsonl",
        type=Path,
        default=Path("webarena_trajs.test.jsonl"),
        help="测试集 jsonl 输出路径",
    )
    parser.add_argument(
        "--train_parquet",
        type=Path,
        default=Path("webarena_trajs.train.parquet"),
        help="训练集 parquet 输出路径",
    )
    parser.add_argument(
        "--test_parquet",
        type=Path,
        default=Path("webarena_trajs.test.parquet"),
        help="测试集 parquet 输出路径",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="训练集比例（0-1）",
    )
    return parser.parse_args()


def extract_text_between(text: str, start_key: str, end_key: str) -> str:
    if start_key not in text:
        return ""
    segment = text.split(start_key, 1)[1]
    if end_key and end_key in segment:
        segment = segment.split(end_key, 1)[0]
    return segment.strip()


def parse_user_observation(user_text: str) -> Dict[str, str]:
    # 首选 nnetnav 风格
    observation = extract_text_between(user_text, "OBSERVATION:\n", "\nURL:")
    url_line = extract_text_between(user_text, "URL:", "\nOBJECTIVE:")
    objective = extract_text_between(user_text, "OBJECTIVE:", "\nPREVIOUS ACTIONS:")

    # 兼容 WebArena/real 数据风格（带 CURRENT OBSERVATION/INTERACTION HISTORY）
    if not observation:
        # CURRENT OBSERVATION 后面通常是整段 AXTree，直到文本末尾或下一个大段标题
        alt_obs = extract_text_between(user_text, "CURRENT OBSERVATION:\n", "")
        if not alt_obs:
            # 有些数据没有换行符分隔，尝试不带换行的关键字
            alt_obs = extract_text_between(user_text, "CURRENT OBSERVATION:", "")
        observation = alt_obs or observation

    if not objective:
        # 在 WebArena 风格中，OBJECTIVE 段常常在 INTERACTION HISTORY 之前
        objective = extract_text_between(user_text, "OBJECTIVE:\n", "\nINTERACTION HISTORY:") or \
                    extract_text_between(user_text, "OBJECTIVE:", "\nINTERACTION HISTORY:") or \
                    extract_text_between(user_text, "OBJECTIVE:", "")

    if not url_line:
        # 某些数据不含 URL 段，允许为空
        url_line = ""

    return {
        "observation": observation.strip(),
        "url": url_line.strip(),
        "objective": objective.strip(),
    }


def parse_assistant_response(assistant_text: str) -> Tuple[str, str]:
    reasoning = assistant_text.strip()
    action = ""
    # 1) 优先从五段式结构中提取 ACTION 段
    parts = ["INTERACTION HISTORY SUMMARY", "OBSERVATION DESCRIPTION", "REASON", "ACTION", "OBSERVATION HIGHLIGHT"]
    sections: Dict[str, List[str]] = {}
    current = None
    for line in assistant_text.split("\n"):
        name = line.strip(": ")
        if name in parts and line.strip().endswith(":"):
            current = name
            sections.setdefault(current, [])
        elif current:
            sections[current].append(line.rstrip())
    if "ACTION" in sections and sections["ACTION"]:
        candidate = "\n".join(sections["ACTION"]).strip()
        first_line = candidate.splitlines()[0].strip()
        if first_line:
            action = first_line
            reasoning = sanitize_reason_text(assistant_text.replace(candidate, "")).strip() or assistant_text.strip()
    # 2) 代码围栏作为次选（兼容历史） 
    if not action:
        matches = ACTION_PATTERN.findall(assistant_text)
        # 取最后一个非空围栏内容，避免使用空字符串作为分隔符导致 ValueError: empty separator
        non_empty = None
        for m in reversed(matches or []):
            if m and m.strip():
                non_empty = m.strip()
                break
        if non_empty:
            action = non_empty
            # 仅在分隔符非空时使用 rsplit
            try:
                reasoning = assistant_text.rsplit(non_empty, 1)[0].strip()
            except ValueError:
                # 极端情况下回退保底
                reasoning = assistant_text.strip()
    # 3) 显式 stop[…] 提取
    if not action:
        stop_match = STOP_PATTERN.findall(assistant_text)
        if stop_match:
            action = stop_match[-1].strip()
            reasoning = assistant_text.rsplit(stop_match[-1], 1)[0].strip()
    # 4) 兜底正则（click/type/go_back 等）
    if not action:
        fallback_pattern = re.compile(
            r"(click\s*\[\d+\]"
            r"|type\s*\[\d+\]\s*\[.*?\]\s*\[(?:0|1)\]"
            r"|hover\s*\[\d+\]"
            r"|scroll\s*\[(?:up|down)\]"
            r"|press\s*\[[^\]]+\]"
            r"|goto\s*\[.*?\]"
            r"|go_back"
            r"|go_forward"
            r"|new_tab"
            r"|close_tab"
            r"|tab_focus\s*\[\d+\]"
            r"|stop\s*\[.*?\])",
            flags=re.IGNORECASE | re.DOTALL,
        )
        m = fallback_pattern.search(assistant_text or "")
        if m:
            action = m.group(1).strip()
    return reasoning.strip(), action.strip()


def normalize_action(action: str) -> str:
    if not action:
        return ""
    cleaned = CODE_FENCE.sub("", action).strip()
    return cleaned


def is_stop_action(action: str) -> bool:
    normalized = normalize_action(action).lower()
    return normalized.startswith("stop")


def build_user_prompt(objective: str, observation: str, history: List[Dict[str, str]]) -> str:
    if not history:
        return f"OBJECTIVE:\n{objective}\nCURRENT OBSERVATION:\n{observation}"
    history_chunks = []
    for idx, entry in enumerate(history):
        obs_short = entry["observation"][:500]
        history_chunks.append(
            f"<step_{idx}_interaction>\nOBSERVATION:\n{obs_short}\n"
            f"REASON FOR ACTION:{entry['reasoning']}\nACTION:\n{entry['action']}\n</step_{idx}_interaction>\n"
        )
    history_str = "".join(history_chunks)
    return (
        f"OBJECTIVE:\n{objective}\nINTERACTION HISTORY:\n{history_str}"
        f"CURRENT OBSERVATION:\n{observation}"
    )


def build_prompt(objective: str, observation: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(objective, observation, history)},
    ]


def summarize_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "No available interaction history."
    chunks = []
    for idx, entry in enumerate(history):
        action = entry.get("action") or "N/A"
        reasoning = entry.get("reasoning") or ""
        reasoning_short = reasoning.replace("\n", " ")[:160]
        chunks.append(f"Step {idx + 1}: Executed {action}. Rationale: {reasoning_short}")
    return " | ".join(chunks)


def extract_action_highlight(action: str) -> str:
    ids = re.findall(r"\[(\d+)\]", action or "")
    if ids:
        return ", ".join(dict.fromkeys(ids))
    return "N/A"


def sanitize_reason_text(text: str) -> str:
    """
    清洗来自原始 assistant 的 reasoning，去掉代码块/动作总结等噪声，避免污染 REASON 段。
    """
    if not text:
        return ""
    # 去掉代码围栏内的内容
    text = re.sub(r"```[\s\S]*?```", "", text)
    # 去掉包含 In summary 的行
    text = re.sub(r"^.*In summary.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


def extract_title_from_axtree(axtree_txt: str) -> str:
    """
    从 axtree 文本首行提取 RootWebArea 的标题，形如：RootWebArea 'Title ...'
    """
    if not axtree_txt:
        return ""
    first_line = axtree_txt.splitlines()[0]
    m = re.search(r"RootWebArea\s+'([^']+)'", first_line)
    return m.group(1) if m else ""


def build_history_block(history: List[Dict[str, str]]) -> str:
    """
    生成与 user 的 INTERACTION HISTORY 一致的可读文本块，供模型做摘要。
    """
    if not history:
        return "No interaction history."
    parts = []
    for i, h in enumerate(history):
        obs = (h.get("observation") or "")[:500]
        rsn = h.get("reasoning") or ""
        act = h.get("action") or "N/A"
        parts.append(
            f"<step_{i}_interaction>\nOBSERVATION:\n{obs}\nREASON FOR ACTION:{rsn}\nACTION:\n{act}\n</step_{i}_interaction>\n"
        )
    return "".join(parts)


def heuristic_observation_desc(objective: str, axtree_txt: str) -> str:
    """
    非模型场景下的观察描述启发式生成：报告标题、与目标相关的高层信息，避免直接贴 axtree。
    """
    title = extract_title_from_axtree(axtree_txt)
    title_part = f"The current page is '{title}'." if title else "The current page is a web view."
    # 简单提取一些关键信息计数
    link_count = len(re.findall(r"\blink\b", axtree_txt[:4000], flags=re.IGNORECASE))
    button_count = len(re.findall(r"\bbutton\b", axtree_txt[:4000], flags=re.IGNORECASE))
    tab_count = len(re.findall(r"\btab\b", axtree_txt[:4000], flags=re.IGNORECASE))
    counts_desc = []
    if link_count:
        counts_desc.append(f"{link_count} links")
    if button_count:
        counts_desc.append(f"{button_count} buttons")
    if tab_count:
        counts_desc.append(f"{tab_count} tabs")
    counts_part = f" Key UI elements present: {', '.join(counts_desc)}." if counts_desc else ""
    objective_part = f" The description focuses on elements relevant to the objective: {objective}."
    return f"{title_part}{counts_part}{objective_part}".strip()


def truncate_text(text: str, limit: int = 1500) -> str:
    if not text:
        return "N/A"
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


class AssistantMessageBuilder:
    def __init__(
        self,
        completion_model: Optional[str],
        device: str,
        max_new_tokens: int,
        temperature: float,
        trust_remote_code: bool,
        device_map: str,
        vllm_base_url: str = "",
        vllm_api_key: str = "",
        vllm_model: str = "",
        vllm_request_timeout: float = 120.0,
    ) -> None:
        self.temperature = max(0.0, temperature)
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.model_name = completion_model
        self.trust_remote_code = trust_remote_code
        self.device_map = (device_map or "").strip().lower()
        self.model = None
        self.tokenizer = None
        # vLLM OpenAI-compatible server config
        self.vllm_base_url = (vllm_base_url or "").rstrip("/")
        self.vllm_api_key = vllm_api_key or ""
        self.vllm_model = vllm_model or ""
        self.vllm_timeout = float(vllm_request_timeout or 120.0)
        self.use_vllm_api = bool(self.vllm_base_url)
        if completion_model:
            if AutoModelForCausalLM is None or AutoTokenizer is None:
                raise ImportError("需要 transformers 才能加载 completion_model，请先安装 transformers。")
            if torch is None:
                raise ImportError("需要 torch 才能运行 completion_model，请先安装 torch。")
            self._load_model()

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code
        )
        if self.device_map and self.device_map != "none":
            # shard across visible devices automatically
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                device_map=self.device_map,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            self.model.to(self.device)
        self.model.eval()

    def build_assistant(
        self,
        user_prompt: str,
        history_entries: List[Dict[str, str]],
        current_observation: str,
        target_action: str,
        reasoning_hint: str,
    ) -> str:
        if self.use_vllm_api:
            assistant_full = self._generate_full_assistant_vllm(user_prompt, history_entries, target_action, reasoning_hint)
            assistant_fixed = self._ensure_action_and_sections(
                assistant_full,
                target_action=target_action,
                history_entries=history_entries,
                current_observation=current_observation,
            )
            return assistant_fixed
        if self.model is None:
            # 无模型：启发式生成观察描述，摘要历史，reason 用提示或默认句
            obs_desc = heuristic_observation_desc("", current_observation)
            hist_summary = summarize_history(history_entries)
            return self._fallback_template(
                history_entries,
                current_observation,
                target_action,
                reasoning_hint,
                observation_desc_text=obs_desc,
                history_summary_text=hist_summary,
            )
        # 有模型：单次调用生成完整 assistant（五段），随后强制 ACTION = 标准动作，必要时回退/修补
        assistant_full = self._generate_full_assistant(user_prompt, history_entries, target_action, reasoning_hint)
        assistant_fixed = self._ensure_action_and_sections(
            assistant_full,
            target_action=target_action,
            history_entries=history_entries,
            current_observation=current_observation,
        )
        return assistant_fixed

    def _generate_full_assistant(
        self,
        user_prompt: str,
        history_entries: List[Dict[str, str]],
        target_action: str,
        reasoning_hint: str,
    ) -> str:
        assert self.model is not None and self.tokenizer is not None
        hist_block = build_history_block(history_entries)
        instruction = (
            "You will receive the exact system+user prompt used during training.\n"
            "Generate ONE assistant message with EXACTLY the following five sections in this order (headings must match EXACTLY):\n"
            "INTERACTION HISTORY SUMMARY:\n"
            "OBSERVATION DESCRIPTION:\n"
            "REASON:\n"
            "ACTION:\n"
            "OBSERVATION HIGHLIGHT:\n\n"
            "CRITICAL RULES:\n"
            f"- The ACTION line must be EXACTLY: {target_action} (do NOT change or rephrase).\n"
            "- Do NOT add any extra sections or headings beyond the five above.\n"
            "- OBSERVATION HIGHLIGHT must be ONLY a comma-separated list of numerical element ids (e.g., 1321, 52, 756). If unsure, leave it empty.\n"
            "- Be concise and strictly base your content on the provided OBJECTIVE, INTERACTION HISTORY and CURRENT OBSERVATION in the user prompt.\n"
            "- In REASON, justify why the specified ACTION is appropriate now.\n"
            "- If there is no prior interaction history, set INTERACTION HISTORY SUMMARY to exactly: No interactions have been performed yet.\n"
        )
        model_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\n{instruction}"
        inputs = self.tokenizer(model_prompt, return_tensors="pt")
        if self.device_map and self.device_map != "none":
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        completion = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return completion

    def _chat_vllm(self, messages: List[Dict[str, str]]) -> str:
        if not self.use_vllm_api:
            return ""
        if requests is None:
            raise ImportError("需要 requests 库以调用 vLLM OpenAI 兼容服务，请先安装 requests。")
        base = self.vllm_base_url
        if not base.endswith("/v1"):
            base = base.rstrip("/") + "/v1"
        endpoint = base + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.vllm_api_key}"
        payload = {
            "model": self.vllm_model or "model",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.vllm_timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content") or ""
        return str(content).strip()

    def _generate_full_assistant_vllm(
        self,
        user_prompt: str,
        history_entries: List[Dict[str, str]],
        target_action: str,
        reasoning_hint: str,
    ) -> str:
        hist_block = build_history_block(history_entries)
        instruction = (
            "You will receive the exact system+user prompt used during training.\n"
            "Generate ONE assistant message with EXACTLY the following five sections in this order (headings must match EXACTLY):\n"
            "INTERACTION HISTORY SUMMARY:\n"
            "OBSERVATION DESCRIPTION:\n"
            "REASON:\n"
            "ACTION:\n"
            "OBSERVATION HIGHLIGHT:\n\n"
            "CRITICAL RULES:\n"
            f"- The ACTION line must be EXACTLY: {target_action} (do NOT change or rephrase).\n"
            "- Do NOT add any extra sections or headings beyond the five above.\n"
            "- OBSERVATION HIGHLIGHT must be ONLY a comma-separated list of numerical element ids (e.g., 1321, 52, 756). If unsure, leave it empty.\n"
            "- Be concise and strictly base your content on the provided OBJECTIVE, INTERACTION HISTORY and CURRENT OBSERVATION in the user prompt.\n"
            "- In REASON, justify why the specified ACTION is appropriate now.\n"
            "- If there is no prior interaction history, set INTERACTION HISTORY SUMMARY to exactly: No interactions have been performed yet.\n"
        )
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        user_msg = {"role": "user", "content": f"{user_prompt}\n\n{instruction}"}
        return self._chat_vllm([system_msg, user_msg])

    def _ensure_action_and_sections(
        self,
        assistant_text: str,
        target_action: str,
        history_entries: List[Dict[str, str]],
        current_observation: str,
    ) -> str:
        """
        - 强制替换 ACTION 段为标准动作
        - 若缺少任一段落，则用启发式/空占位补齐，确保五段名称与顺序正确
        """
        text = assistant_text or ""
        parts = ["INTERACTION HISTORY SUMMARY", "OBSERVATION DESCRIPTION", "REASON", "ACTION", "OBSERVATION HIGHLIGHT"]

        # 简单分段解析
        sections: dict[str, str] = {}
        current = None
        lines = text.split("\n")
        for line in lines:
            name = line.strip(": ")
            if name in parts and line.strip().endswith(":"):
                current = name
                sections.setdefault(current, [])
            elif current:
                sections[current].append(line.rstrip())

        # 组装字符串内容
        def join_sec(name: str) -> str:
            return "\n".join(sections.get(name, [])).strip()

        # 补齐/修正各段
        if "INTERACTION HISTORY SUMMARY" not in sections or not join_sec("INTERACTION HISTORY SUMMARY"):
            sections["INTERACTION HISTORY SUMMARY"] = [summarize_history(history_entries)]
        if "OBSERVATION DESCRIPTION" not in sections or not join_sec("OBSERVATION DESCRIPTION"):
            sections["OBSERVATION DESCRIPTION"] = [heuristic_observation_desc("", current_observation)]
        if "REASON" not in sections or not join_sec("REASON"):
            sections["REASON"] = ["Based on the current observation and history, execute the action to move toward the objective."]
        # ACTION 强制为标准动作
        sections["ACTION"] = [target_action]
        # HIGHLIGHT 保底用动作解析
        if "OBSERVATION HIGHLIGHT" not in sections or not join_sec("OBSERVATION HIGHLIGHT"):
            sections["OBSERVATION HIGHLIGHT"] = [extract_action_highlight(target_action)]

        # 重新按顺序渲染
        rendered = []
        for name in parts:
            rendered.append(f"{name}:\n{join_sec(name)}")
        return "\n\n".join(rendered)

    def _fallback_template(
        self,
        history_entries: List[Dict[str, str]],
        current_observation: str,
        target_action: str,
        reasoning_hint: str,
        observation_desc_text: str | None = None,
        history_summary_text: str | None = None,
        highlight_text: str | None = None,
    ) -> str:
        history_summary = history_summary_text or summarize_history(history_entries)
        observation_desc = observation_desc_text or truncate_text(current_observation)
        reasoning_text = reasoning_hint.strip() if reasoning_hint else "Based on the current observation and history, execute the action to move toward the objective."
        highlight = (highlight_text or "").strip() or extract_action_highlight(target_action)
        return (
            "INTERACTION HISTORY SUMMARY:\n"
            f"{history_summary}\n\n"
            "OBSERVATION DESCRIPTION:\n"
            f"{observation_desc}\n\n"
            "REASON:\n"
            f"{reasoning_text}\n\n"
            "ACTION:\n"
            f"{target_action or 'stop [N/A]'}\n\n"
            "OBSERVATION HIGHLIGHT:\n"
            f"{highlight}\n"
        )

    def _generate_reason_only(self, user_prompt: str, target_action: str, reasoning_hint: str) -> str:
        assert self.model is not None and self.tokenizer is not None
        instruction = (
            "You will receive the exact system+user prompt used during training.\n"
            "We have already determined the standard action (DO NOT output this action; for reference only):\n"
            f"{target_action or 'stop [N/A]'}\n\n"
            "Please output ONLY the REASON paragraph as plain text (no headings; DO NOT output ACTION/OBSERVATION/HISTORY or any other sections).\n"
            "The REASON should explain why this action is appropriate given the current observation and the interaction history. You may use the following hint:"
        )
        hint = reasoning_hint or "No extra hint."
        model_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\n{instruction}\n{hint}"
        inputs = self.tokenizer(model_prompt, return_tensors="pt")
        # If using device_map='auto', place inputs on cuda:0 when available; otherwise on model device
        if self.device_map and self.device_map != "none":
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        completion = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not completion:
            completion = hint if isinstance(hint, str) and hint.strip() else "Based on the current observation and history, execute the action to move toward the objective."
        return completion

    def _generate_obs_desc_only(self, user_prompt: str) -> str:
        assert self.model is not None and self.tokenizer is not None
        instruction = (
            "You will receive the exact system+user prompt used during training.\n"
            "Please output ONLY the OBSERVATION DESCRIPTION paragraph as plain text (no headings; DO NOT output ACTION/REASON/HISTORY or any other sections).\n"
            "Focus on salient visual/structural cues that are relevant to the objective."
        )
        model_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\n{instruction}"
        inputs = self.tokenizer(model_prompt, return_tensors="pt")
        if self.device_map and self.device_map != "none":
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        completion = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return completion

    def _generate_history_summary_only(self, history_block: str, objective: str) -> str:
        assert self.model is not None and self.tokenizer is not None
        prompt = (
            "You will receive an objective and an interaction history block.\n"
            "Please output ONLY a concise summary paragraph emphasizing the most important details from the history relevant to the objective.\n"
            "Do not include headings or extra sections.\n\n"
            f"OBJECTIVE:\n{objective}\n\nINTERACTION HISTORY:\n{history_block}"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device_map and self.device_map != "none":
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        completion = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return completion

    def _generate_highlight_only(self, user_prompt: str, target_action: str) -> str:
        """
        让模型仅输出当前观察相关的元素 id 列表（逗号分隔），用于 OBSERVATION HIGHLIGHT。
        """
        assert self.model is not None and self.tokenizer is not None
        instruction = (
            "You will receive the exact system+user prompt used during training.\n"
            "Please output ONLY a comma-separated list of numerical element ids from the CURRENT OBSERVATION that justify the given action.\n"
            "Do not include any text other than numbers and commas.\n"
            "If unsure, return an empty string.\n\n"
            f"The action to justify is: {target_action}"
        )
        model_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\n{instruction}"
        inputs = self.tokenizer(model_prompt, return_tensors="pt")
        if self.device_map and self.device_map != "none":
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": 64,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        completion = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        # 只保留数字和逗号
        completion = re.sub(r"[^0-9,]", "", completion)
        return completion.strip(", ")


def load_nnetnav_records(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            task_id = sample.get("task_name") or sample.get("id")
            grouped.setdefault(task_id, []).append(sample)
    return grouped


def convert_task_steps(task_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    steps = []
    for idx, sample in enumerate(task_samples):
        messages = sample["messages"]
        user_text = messages[1]["content"]
        assistant_text = messages[-1]["content"]
        user_info = parse_user_observation(user_text)
        reasoning, action = parse_assistant_response(assistant_text)
        steps.append(
            {
                "objective": user_info["objective"],
                "observation": user_info["observation"],
                "url": user_info["url"],
                "assistant_reasoning": reasoning,
                "action": action,
                "assistant_full": assistant_text,
                "step_index": idx,
            }
        )
    for i in range(len(steps) - 1):
        steps[i]["next_observation"] = steps[i + 1]["observation"]
    if steps:
        steps[-1]["next_observation"] = ""
    return steps


def build_window_samples(
    task_id: str,
    steps: List[Dict[str, Any]],
    dataset_name: str,
    rng: random.Random,
    assistant_builder: AssistantMessageBuilder,
) -> List[Dict[str, Any]]:
    if len(steps) < 2:
        return []
    max_len = min(5, len(steps))
    lengths = list(range(2, max_len + 1))
    rng.shuffle(lengths)
    selected_lengths = sorted(lengths[: min(2, len(lengths))])
    samples = []
    for length in selected_lengths:
            window = steps[-length:]
            # 以窗口起点为起始，逐步滚动到最终 stop；每一步都生成一条 assistant 并拼接成完整对话
            # 要求：每步对话三段结构且 system 要重复（system→user→assistant）
            multi_step_messages: List[Dict[str, str]] = []
            rolling_history: List[Dict[str, str]] = []
            objective = window[0]["objective"]
            url_at_start = window[0]["url"]

            for idx, cur_step in enumerate(window):
                user_content = build_user_prompt(
                    objective=objective,
                    observation=cur_step["observation"],
                    history=rolling_history,
                )
                assistant_content = assistant_builder.build_assistant(
                    user_prompt=user_content,
                    history_entries=rolling_history,
                    current_observation=cur_step["observation"],
                    target_action=cur_step["action"],
                    reasoning_hint=sanitize_reason_text(cur_step["assistant_reasoning"]),
                )
                multi_step_messages.append({"role": "system", "content": SYSTEM_PROMPT})
                multi_step_messages.append({"role": "user", "content": user_content})
                multi_step_messages.append({"role": "assistant", "content": assistant_content})

                # 将当前步加入滚动历史，用于下一步的 INTERACTION HISTORY
                # 使用 assistant 五段式中的 OBSERVATION DESCRIPTION 与 REASON 作为历史
                parts = ["INTERACTION HISTORY SUMMARY", "OBSERVATION DESCRIPTION", "REASON", "ACTION", "OBSERVATION HIGHLIGHT"]
                sections: Dict[str, List[str]] = {}
                cur_name = None
                for line in (assistant_content or "").split("\n"):
                    nm = line.strip(": ")
                    if nm in parts and line.strip().endswith(":"):
                        cur_name = nm
                        sections.setdefault(cur_name, [])
                    elif cur_name:
                        sections[cur_name].append(line.rstrip())
                obs_desc = "\n".join(sections.get("OBSERVATION DESCRIPTION", [])).strip()
                rsn_text = "\n".join(sections.get("REASON", [])).strip()
                rolling_history.append(
                    {
                        "observation": (obs_desc or cur_step["observation"])[:2000],
                        "reasoning": (rsn_text or sanitize_reason_text(cur_step["assistant_reasoning"]))[:2000],
                        "action": cur_step["action"] or "N/A",
                    }
                )

            task_block = {
                "objective": objective,
                "axtree_txt": window[0]["observation"],
                "url": url_at_start,
            }
            extra_info = {
                "task": task_block,
                "trajectory_id": task_id,
                "window_length": length,
                "window_start_step": window[0]["step_index"],
                "window_end_step": window[-1]["step_index"],
            }
            # 兼容训练字段：保留 prompt 为起点的 system+user；标准答案轨迹放在 standard_answer_messages
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(objective, window[0]["observation"], [])},
            ]
            samples.append(
                {
                    "data_source": dataset_name,
                    "prompt": prompt,
                    "ability": "web",
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": extra_info,
                    "task": task_block,
                    "standard_answer_messages": multi_step_messages,
                }
            )
    return samples

def main() -> None:
    args = parse_args()
    rng = random.Random(args.random_seed)
    assistant_builder = AssistantMessageBuilder(
        completion_model=args.completion_model,
        device=args.completion_device,
        max_new_tokens=args.completion_max_new_tokens,
        temperature=args.completion_temperature,
        trust_remote_code=args.completion_trust_remote_code,
        device_map=args.device_map,
        vllm_base_url=args.vllm_base_url,
        vllm_api_key=args.vllm_api_key,
        vllm_model=args.vllm_model,
        vllm_request_timeout=args.vllm_request_timeout,
    )
    grouped = load_nnetnav_records(args.input_jsonl)
    dataset_name = args.input_jsonl.stem
    all_samples: List[Dict[str, Any]] = []

    def _process_one(task_id: str, samples: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
        local_rng = random.Random((seed * 1315423911) ^ hash(task_id))
        steps = convert_task_steps(samples)
        if not steps:
            return []
        if not is_stop_action(steps[-1]["action"]):
            return []
        return build_window_samples(task_id, steps, dataset_name, local_rng, assistant_builder)

    if max(1, int(args.num_workers)) > 1:
        with ThreadPoolExecutor(max_workers=max(1, int(args.num_workers))) as ex:
            futures = [ex.submit(_process_one, task_id, samples, args.random_seed) for task_id, samples in grouped.items()]
            for fut in as_completed(futures):
                try:
                    ws = fut.result()
                    if ws:
                        all_samples.extend(ws)
                except Exception as e:
                    print(f"[warn] group failed: {e}")
    else:
        for task_id, samples in grouped.items():
            ws = _process_one(task_id, samples, args.random_seed)
            if ws:
                all_samples.extend(ws)

    if not all_samples:
        print("No samples generated.")
        return

    # Shuffle and split
    rng.shuffle(all_samples)
    split_idx = int(len(all_samples) * max(0.0, min(1.0, args.train_ratio)))
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]

    # Mark split for convenience
    for r in train_samples:
        r.setdefault("extra_info", {}).update({"split": "train"})
    for r in test_samples:
        r.setdefault("extra_info", {}).update({"split": "test"})

    # Write JSONL（并打印绝对路径与回读校验行数）
    args.train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    train_path = args.train_jsonl.resolve()
    with train_path.open("w", encoding="utf-8") as f:
        for record in train_samples:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # verify
    try:
        with train_path.open("r", encoding="utf-8") as f:
            verify_lines = sum(1 for _ in f)
        print(f"[+] Wrote {len(train_samples)} train samples to {train_path} (verify_lines={verify_lines})")
    except Exception as e:
        print(f"[warn] verify train jsonl failed: {e}")

    args.test_jsonl.parent.mkdir(parents=True, exist_ok=True)
    test_path = args.test_jsonl.resolve()
    with test_path.open("w", encoding="utf-8") as f:
        for record in test_samples:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    try:
        with test_path.open("r", encoding="utf-8") as f:
            verify_lines = sum(1 for _ in f)
        print(f"[+] Wrote {len(test_samples)} test samples to {test_path} (verify_lines={verify_lines})")
    except Exception as e:
        print(f"[warn] verify test jsonl failed: {e}")

    # Write Parquet (optional)
    if pd is not None:
        if args.train_parquet:
            args.train_parquet.parent.mkdir(parents=True, exist_ok=True)
            train_parquet_path = args.train_parquet.resolve()
            pd.DataFrame(train_samples).to_parquet(train_parquet_path, index=False)
            print(f"[+] Wrote train parquet to {train_parquet_path}")
        if args.test_parquet:
            args.test_parquet.parent.mkdir(parents=True, exist_ok=True)
            test_parquet_path = args.test_parquet.resolve()
            pd.DataFrame(test_samples).to_parquet(test_parquet_path, index=False)
            print(f"[+] Wrote test parquet to {test_parquet_path}")


if __name__ == "__main__":
    main()

