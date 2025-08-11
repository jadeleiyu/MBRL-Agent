# ==============================================
# File: encoder.py
# ==============================================
"""
Hybrid web state encoder.

Converts raw HTML into a compact, task-aware JSON state containing:
  - header: {url, title, instruction, summary}
  - candidates: top-K interactable elements with attributes, short text, and neighborhood
  - (optional) diff vs previous state to minimize tokens at step > 0

Design goals:
  • Keep tokens small (~1-2k) while preserving element-level grounding.
  • Stable pointers via `id` + `selector` so actions can be executed reliably.
  • Deterministic, fast; optionally plug an LLM summarizer for the header.

Dependencies: bs4 (BeautifulSoup), lxml, scikit-learn
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# -----------------------------
# Prompt templates (for LLM-based encoding if desired)
# -----------------------------
STATE_ENCODER_SYSTEM_PROMPT = (
    """You are a WEB STATE ENCODER for a text-only web agent.\n"
    "Given an instruction, URL, and RAW HTML of a page, produce a COMPACT JSON that the agent will read.\n"
    "Return STRICT JSON with keys: {\n"
    "  \"url\": str, \n"
    "  \"title\": str, \n"
    "  \"instruction\": str, \n"
    "  \"summary\": str,  // 1–3 sentences, high-level sections and controls\n"
    "  \"candidates\": [  // top-K interactable elements\n"
    "    {\"id\": str, \"tag\": str, \"role\": str, \"text\": str, \n"
    "     \"attrs\": {id?, name?, class?, placeholder?, aria-label?, data-testid?, role?},\n"
    "     \"selector\": str,  // stable CSS selector when possible\n"
    "     \"pos\": {\"depth\": int, \"order\": int, \"visible\": bool},\n"
    "     \"neighborhood\": str   // 1–2 lines: labels, item title/price/options\n"
    "    }\n"
    "  ],\n"
    "  \"diff\": {\"added\": [str], \"removed\": [str], \"changed\": [str]}  // optional\n"
    "}.\n"
    "Rules: Keep summary concise; prefer buttons/inputs/links/selects. Include only meaningful attributes.\n"
    "Do not include prose outside JSON.\n"""
)

STATE_ENCODER_USER_PROMPT = (
    """URL: {url}\n"
    "INSTRUCTION: {instruction}\n"
    "RAW_HTML:\n{html}\n"
    "Top-K: {topk}\n"
    "Return JSON now."""
)


# -----------------------------
# Helper utilities
# -----------------------------

_INTERACTABLE_TAGS = {
    "a": 0.9,
    "button": 1.0,
    "input": 1.0,
    "select": 0.95,
    "textarea": 0.95,
    "summary": 0.7,
    "option": 0.5,
}

_ROLE_PRIORS = {
    "button": 1.0,
    "link": 0.9,
    "textbox": 0.95,
    "combobox": 0.95,
    "menuitem": 0.8,
    "tab": 0.8,
}

_VISIBLE_RE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)


def _is_visible(tag: Tag) -> bool:
    if not isinstance(tag, Tag):
        return False
    style = tag.get("style", "")
    if _VISIBLE_RE.search(style or ""):
        return False
    if tag.name in ("script", "style", "template", "noscript"):
        return False
    return True


def _text_of(tag: Tag, limit: int = 160) -> str:
    """Extract human-visible text, trimmed."""
    if not tag:
        return ""
    # Avoid deep recursion cost: take direct text + a bit of children
    txt = " ".join(s for s in tag.stripped_strings)
    return txt[:limit]


def _label_for(tag: Tag) -> str:
    # Prefer aria-label, then label[for], then placeholder, then title
    aria = tag.get("aria-label")
    if aria:
        return aria
    placeholder = tag.get("placeholder")
    if placeholder:
        return placeholder
    title = tag.get("title")
    if title:
        return title
    return ""


def _stable_selector(tag: Tag) -> str:
    # Try id, data-testid, name, aria-label, role+tag, class first token
    def q(s: Optional[str]) -> str:
        if not s:
            return ""
        return s.replace("\\", "\\\\").replace("'", "\\'")

    if tag.get("id"):
        return f"#{q(tag.get('id'))}"
    for key in ("data-testid", "data-test-id", "data_testid"):
        if tag.get(key):
            return f"[data-testid='{q(tag.get(key))}']"
    if tag.get("name"):
        return f"[name='{q(tag.get('name'))}']"
    if tag.get("aria-label"):
        return f"[aria-label='{q(tag.get('aria-label'))}']"
    cls = tag.get("class")
    if isinstance(cls, list) and cls:
        return "." + q(cls[0])
    role = tag.get("role")
    if role:
        return f"{tag.name}[role='{q(role)}']"
    # fallback: tag + nth-child along ancestors (short)
    path = []
    cur = tag
    for _ in range(3):  # last 3 ancestors
        if not cur or not isinstance(cur, Tag):
            break
        sibs = [c for c in cur.parent.children if isinstance(c, Tag) and c.name == cur.name] if cur.parent else []
        if sibs and len(sibs) > 1:
            idx = sibs.index(cur) + 1
            path.append(f"{cur.name}:nth-of-type({idx})")
        else:
            path.append(cur.name)
        cur = cur.parent
    return " > ".join(reversed(path))


def _stable_id(tag: Tag) -> str:
    for key in ("id", "data-testid", "name"):
        if tag.get(key):
            return str(tag.get(key))
    # Hash a short DOM path as fallback
    path = []
    cur = tag
    while cur and isinstance(cur, Tag) and len(path) < 6:
        sibs = [c for c in cur.parent.children if isinstance(c, Tag) and c.name == cur.name] if cur.parent else []
        idx = sibs.index(cur) if sibs else 0
        path.append(f"{cur.name}:{idx}")
        cur = cur.parent
    digest = hashlib.md5("/".join(path).encode("utf-8")).hexdigest()[:10]
    return f"auto-{digest}"


def _depth(tag: Tag) -> int:
    d, cur = 0, tag
    while cur and isinstance(cur, Tag):
        d += 1
        cur = cur.parent
    return d


def _reading_order_index(tag: Tag) -> int:
    # approximate by enumerating document order
    idx = 0
    node = tag
    while node and node.previous_sibling is not None:
        node = node.previous_sibling
        if isinstance(node, Tag):
            idx += 1
    return idx


def _attrs_subset(tag: Tag) -> Dict[str, Any]:
    keep = [
        "id",
        "name",
        "class",
        "placeholder",
        "aria-label",
        "role",
        "href",
        "value",
        "type",
        "data-testid",
    ]
    out: Dict[str, Any] = {}
    for k in keep:
        v = tag.get(k)
        if v:
            out[k] = v if isinstance(v, (str, int, float)) else (v[0] if isinstance(v, list) and v else None)
    return out


def _neighborhood(tag: Tag, limit: int = 140) -> str:
    texts: List[str] = []
    # parent label
    p = tag.parent if tag else None
    if isinstance(p, Tag):
        label_like = []
        for sib in p.children:
            if isinstance(sib, Tag) and sib is not tag:
                t = _text_of(sib, 60)
                if t:
                    label_like.append(t)
        if label_like:
            texts.append(" | ".join(label_like[:3]))
    # up to item title / price from ancestors
    cur = tag
    for _ in range(2):
        cur = cur.parent if isinstance(cur, Tag) else None
        if isinstance(cur, Tag):
            t = _text_of(cur, 60)
            if t:
                texts.append(t)
    s = " | ".join([t for t in texts if t])
    return s[:limit]


def _structure_prior(tag: Tag) -> float:
    base = _INTERACTABLE_TAGS.get(tag.name, 0.4)
    role = tag.get("role")
    if role:
        base = max(base, _ROLE_PRIORS.get(role, 0.4))
    # clickable links with href
    if tag.name == "a" and tag.get("href"):
        base = max(base, 0.85)
    return float(base)


@dataclass
class EncodedState:
    url: str
    title: str
    instruction: str
    summary: str
    candidates: List[Dict[str, Any]]
    diff: Optional[Dict[str, List[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "url": self.url,
            "title": self.title,
            "instruction": self.instruction,
            "summary": self.summary,
            "candidates": self.candidates,
        }
        if self.diff is not None:
            d["diff"] = self.diff
        return d


class WebStateEncoder:
    def __init__(
        self,
        top_k: int = 30,
        neighborhood_chars: int = 140,
        include_diff: bool = True,
    ):
        self.top_k = top_k
        self.neighborhood_chars = neighborhood_chars
        self.include_diff = include_diff

    # ---------- public API ----------
    def encode(self, html: str, instruction: str, url: str, prev_state: Optional[Dict[str, Any]] = None) -> EncodedState:
        soup = BeautifulSoup(html or "", "lxml")
        title = (soup.title.get_text(strip=True) if soup.title else "").strip()
        # 1) collect interactable elements
        elements = self._gather_candidates(soup)
        # 2) score & select top-K
        ranked = self._rank(elements, instruction)
        top = ranked[: self.top_k]
        # 3) build candidate dicts
        cands = [self._to_candidate_dict(tag, i) for i, tag in enumerate(top)]
        # 4) header summary (rule-based)
        summary = self._summarize(soup)
        # 5) diff vs prev
        diff = None
        if self.include_diff and prev_state and isinstance(prev_state, dict) and prev_state.get("candidates"):
            diff = self._diff(prev_state.get("candidates", []), cands)
        return EncodedState(url=url, title=title, instruction=instruction, summary=summary, candidates=cands, diff=diff)

    # ---------- internals ----------
    def _gather_candidates(self, soup: BeautifulSoup) -> List[Tag]:
        tags = []
        for tag in soup.find_all(True):
            if not _is_visible(tag):
                continue
            if tag.name in _INTERACTABLE_TAGS or tag.has_attr("onclick") or tag.get("href"):
                tags.append(tag)
        return tags

    def _rank(self, tags: List[Tag], instruction: str) -> List[Tag]:
        if not tags:
            return []
        # Build documents for tf-idf: instruction + each tag's context
        docs = [instruction]
        contexts = []
        for t in tags:
            text = _text_of(t, 80)
            lab = _label_for(t)
            neigh = _neighborhood(t, self.neighborhood_chars)
            attrs = _attrs_subset(t)
            ctx = " ".join(filter(None, [text, lab, json.dumps(attrs, ensure_ascii=False), neigh]))
            contexts.append(ctx)
            docs.append(ctx)
        vec = TfidfVectorizer(min_df=1, max_features=4096)
        X = vec.fit_transform(docs)
        q = X[0]
        M = X[1:]
        # cosine sims
        denom = (np.linalg.norm(q.toarray()) + 1e-8) * (np.linalg.norm(M.toarray(), axis=1) + 1e-8)
        sims = (M @ q.T).toarray().ravel() / denom
        sims = np.nan_to_num(sims)
        # structure prior & position prior
        struct = np.array([_structure_prior(t) for t in tags])
        pos = np.array([1.0 / (1 + _depth(t)) for t in tags])  # shallower preferred
        # combine (weights can be tuned)
        scores = 2.0 * sims + 1.0 * struct + 0.5 * pos
        order = np.argsort(-scores)
        ranked = [tags[i] for i in order]
        return ranked

    def _to_candidate_dict(self, tag: Tag, order_idx: int) -> Dict[str, Any]:
        attrs = _attrs_subset(tag)
        return {
            "id": _stable_id(tag),
            "tag": tag.name,
            "role": attrs.get("role", ""),
            "text": _text_of(tag, 120),
            "attrs": attrs,
            "selector": _stable_selector(tag),
            "pos": {"depth": _depth(tag), "order": order_idx, "visible": True},
            "neighborhood": _neighborhood(tag, self.neighborhood_chars),
        }

    def _summarize(self, soup: BeautifulSoup) -> str:
        # very lightweight, deterministic summary
        counts = {
            "forms": len(soup.find_all("form")),
            "inputs": len(soup.find_all(["input", "textarea", "select"])),
            "buttons": len(soup.find_all("button")),
            "links": len(soup.find_all("a")),
            "tables": len(soup.find_all("table")),
            "lists": len(soup.find_all(["ul", "ol"]))
        }
        parts = []
        if counts["forms"] or counts["inputs"]:
            parts.append(f"Form elements present ({counts['inputs']} inputs/selects).")
        if counts["buttons"]:
            parts.append(f"{counts['buttons']} buttons.")
        if counts["links"]:
            parts.append(f"{counts['links']} links.")
        if counts["tables"]:
            parts.append("Table(s) detected.")
        if counts["lists"]:
            parts.append("List or grid of items.")
        if not parts:
            parts = ["No obvious interactive controls detected."]
        return " ".join(parts)[:240]

    def _diff(self, prev_cands: List[Dict[str, Any]], cur_cands: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        prev_ids = {c.get("id") for c in prev_cands}
        cur_ids = {c.get("id") for c in cur_cands}
        added = sorted(list(cur_ids - prev_ids))
        removed = sorted(list(prev_ids - cur_ids))
        changed = []
        prev_map = {c.get("id"): c for c in prev_cands}
        for c in cur_cands:
            pid = c.get("id")
            if pid in prev_map and (c.get("text") != prev_map[pid].get("text") or json.dumps(c.get("attrs")) != json.dumps(prev_map[pid].get("attrs"))):
                changed.append(pid)
        return {"added": added, "removed": removed, "changed": sorted(list(set(changed)))}


# -----------------------------
# Convenience: render state to a compact prompt block for the policy
# -----------------------------

def render_state_for_policy(state: Dict[str, Any]) -> str:
    """Pretty-prints the hybrid state into a short text block suitable for prompting.
    Keep it deterministic and compact.
    """
    header = (
        f"URL: {state.get('url','')}\n"
        f"Title: {state.get('title','')}\n"
        f"Instruction: {state.get('instruction','')}\n"
        f"Summary: {state.get('summary','')}\n"
    )
    lines = [header, "Candidates (top-K):"]
    for c in state.get("candidates", [])[: state.get("top_k", 30)]:
        line = (
            f"- ID={c.get('id')} TAG={c.get('tag')} ROLE={c.get('role','')} SEL={c.get('selector')}\n"
            f"  TEXT={c.get('text','')[:80]} | NEIGHBOR={c.get('neighborhood','')[:100]}"
        )
        lines.append(line)
    if state.get("diff"):
        lines.append(f"Diff: {json.dumps(state['diff'])}")
    return "\n".join(lines)
