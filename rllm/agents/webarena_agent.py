#!/usr/bin/env python3

import base64
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from browsergym.utils.obs import _process_bid
from PIL import Image

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.system_prompts import *

logger = logging.getLogger(__name__)

IGNORED_AXTREE_ROLES = ["LineBreak"]

IGNORED_AXTREE_PROPERTIES = (
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
    "focusable",
    "focused",
    "autocomplete",
    "hasPopup",
    "expanded",
    "multiselectable",
    "orientation",
    "controls",
)

UNINTERACTIVE_ROLES = [
    "StaticText",
    "LabelText",
    "main",
    "heading",
    "LayoutTable",
    "tabpanel",
    "LayoutTableRow",
    "LayoutTableCell",
    "time",
    "list",
    "contentinfo",
    "table",
    "row",
    "rowheader",
    "columnheader",
    "gridcell",
    "caption",
    "DescriptionList",
    "DescriptionListTerm",
    "DescriptionListDetail",
    "RootWebArea",
    "rowgroup",
    "alert",
    "cell",  # TODO: should this be non-interactive?
    "code",
]
ROLE_REMAP = {
    "StaticText": "text",
    "LabelText": "text",
    # "caption": "text",
    # "generic": "text"
}

# TODO: need list and table reformating


@dataclass
class Node:
    node_id: int
    role: str
    name: str | None = None
    value: Any = None
    properties: list[str | dict[str, Any]] = field(default_factory=list)
    parent: Optional["Node"] = None
    children: list["Node"] = field(default_factory=list)
    bid: int | None = None
    # visible: bool = True

    def get_all_siblings(self) -> list["Node"]:
        parent = self.parent
        if parent is None:
            return []
        return parent.children


def from_axtree_construct_tree(tree_dict: dict, return_id2node: bool = False) -> Node | tuple[Node, dict[Any, Node]]:
    assert isinstance(tree_dict, dict) and "nodes" in tree_dict
    nodes = tree_dict["nodes"]
    node_id_to_idx = {}
    for idx, node in enumerate(nodes):
        node_id_to_idx[node["nodeId"]] = idx

    if return_id2node:
        id2node: dict[Any, Node] = {}

    def dfs(d: dict) -> Node:
        name = d["name"]["value"] if "name" in d else None
        role = d["role"]["value"]
        properties = d.get("properties", [])
        value = d["value"].get("value", None) if "value" in d else None
        bid = d.get("browsergym_id", None)
        cur_node = Node(
            node_id=d["nodeId"],
            role=role,
            name=name,
            value=value,
            properties=properties,
            bid=bid,
        )
        if "browsergym_id" in d and return_id2node:
            id2node[d["browsergym_id"]] = cur_node

        for child_node_id in d["childIds"]:
            if child_node_id not in node_id_to_idx or child_node_id == d["nodeId"]:
                continue
            child_d = dfs(nodes[node_id_to_idx[child_node_id]])
            child_d.parent = cur_node
            cur_node.children.append(child_d)

        return cur_node

    root = dfs(nodes[0])
    if return_id2node:
        return root, id2node
    return root


def remove_unwanted_characters(text: str):
    """
    Keeps all alphanumeric characters and basic punctuation
    Additionally preserves:
    Currency symbols ($)
    Percentage sign (%)
    Mathematical operators (+, -, =, *)
    HTML/XML brackets (<, >, [, ], {, })
    Other useful web symbols (#, ~, ^, |, \\, °)
    This will ensure that important information like:

    Prices (e.g., "$19.99")
    Percentages (e.g., "25%")
    Mathematical expressions (e.g., "2 + 2 = 4")
    HTML/XML tags
    Temperature (e.g., "72°F")
    URLs and paths (with all special characters)
    are preserved while still removing truly unwanted characters like various Unicode control characters or formatting characters that don't add value to the text content.
    """
    text = text.replace("\xa0", " ")
    cleaned_text = re.sub(
        r"[^\w\s,.!?;:\-\'\"()&/\u2019@$%+=*#~`^<>[\]{}|\\°\n\t]+",
        "",
        text,
        flags=re.UNICODE,
    )
    return cleaned_text


def clean_accesibility_tree(root: Node) -> Node:
    assert isinstance(root, Node)
    if isinstance(root.value, str):
        root.value = remove_unwanted_characters(root.value)
    if isinstance(root.name, str):
        root.name = remove_unwanted_characters(root.name)
    for prop in root.properties:
        if isinstance(prop, dict):
            for key, value in prop.items():
                if isinstance(value, str):
                    prop[key] = remove_unwanted_characters(value)
    for child in root.children:
        clean_accesibility_tree(child)
    return root


def prune_axtree(
    root: Node,
    extra_properties: dict | None = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    skip_generic: bool = True,
    filter_visible_only: bool = False,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
    ignored_roles=IGNORED_AXTREE_ROLES,
    ignored_properties=IGNORED_AXTREE_PROPERTIES,
    remove_redundant_static_text: bool = True,
    hide_all_children: bool = False,
    remove_redundant_mode: str = "parent",  # set to "ancestor" might be too aggressive for code rendering, but good for other websites
    merge_consecutive_static_text: bool = True,
    remove_sibling_with_duplicate_name: bool = False,  # this might add some error into code rendering, but good for other websites
    remove_img_if_child_img: bool = True,
    merge_code_into_text: bool = True,  # replace the code block with markdown format text
) -> list[Node]:
    def dfs(
        node: Node,
        depth: int,
        parent_node_filtered: bool,
        parent_node_name: str,
        ancestor_node_names: str,
    ) -> list[Node]:
        bid = node.bid
        role = node.role
        name = node.name
        skip_node = False
        filter_node = False

        has_attributes = False
        for property in node.properties:
            if not isinstance(property, dict):
                continue
            if "value" not in property or "value" not in property["value"]:
                continue
            prop_name = property["name"]
            prop_value = property["value"]["value"]
            if prop_name in ignored_properties:
                continue
            elif prop_name in ("required", "focused", "atomic"):
                if prop_value:
                    has_attributes = True
            else:
                has_attributes = True

        if role in ignored_roles or name is None:
            skip_node = True

        if (
            role
            in [
                "generic",
                "img",
                "list",
                "strong",
                "paragraph",
                "banner",
                "navigation",
                "Section",
                "LabelText",
                "Legend",
                "listitem",
            ]
            and not (name and name.strip())
            and node.properties == []
        ):
            skip_node = True
        elif role in ["listitem"]:
            skip_node = True
        elif merge_code_into_text and role == "code" and not (name and name.strip()) and node.properties == []:
            skip_node = True
            for child in node.children:
                if child.role == "StaticText" and child.name:
                    child.name = "`" + child.name + "`"

        if skip_generic and role == "generic" and not has_attributes:
            skip_node = True

        if hide_all_children and parent_node_filtered:
            skip_node = True

        if role == "StaticText":
            if parent_node_filtered:
                skip_node = True
            elif remove_redundant_static_text and name and name in parent_node_name:
                skip_node = True
            elif remove_redundant_static_text and remove_redundant_mode == "ancestor" and name and name in ancestor_node_names:
                skip_node = True
        else:
            filter_node, _ = _process_bid(
                bid,
                extra_properties=extra_properties,
                with_visible=with_visible,
                with_clickable=with_clickable,
                with_center_coords=with_center_coords,
                with_bounding_box_coords=with_bounding_box_coords,
                with_som=with_som,
                filter_visible_only=filter_visible_only,
                filter_with_bid_only=filter_with_bid_only,
                filter_som_only=filter_som_only,
                coord_decimals=coord_decimals,
            )

            # if either is True, skip the node
            skip_node = skip_node or filter_node

        child_filtered: list[Node] = []

        for child in node.children:
            name_for_child = name if name else ""
            child_filtered.extend(
                dfs(
                    child,
                    depth if skip_node else depth + 1,
                    filter_node,
                    name_for_child,
                    ancestor_node_names + " " + name_for_child,
                )
            )

        if merge_consecutive_static_text:
            child_after_merge_consecutive_static_text: list[Node] = []
            pre_is_text = False
            for child in child_filtered:
                if child.role == "StaticText":
                    if pre_is_text:
                        # TODO: this does not work well for website contain codes
                        if child_after_merge_consecutive_static_text[-1].name and child.name:
                            child_after_merge_consecutive_static_text[-1].name += " " + child.name
                    else:
                        child_after_merge_consecutive_static_text.append(child)
                    pre_is_text = True
                else:
                    pre_is_text = False
                    child_after_merge_consecutive_static_text.append(child)
            child_filtered = child_after_merge_consecutive_static_text
        # remove static text if there are siblings (might not be static text) have duplicate name
        if remove_sibling_with_duplicate_name:
            child_names = " ".join([child.name for child in child_filtered if child.name])
            child_after_filter_redundant_static_text = []
            for child in child_filtered:
                if not (child.role == "StaticText" and child.name and child_names.count(child.name) > 1):
                    child_after_filter_redundant_static_text.append(child)
            child_filtered = child_after_filter_redundant_static_text

        if node.role == "StaticText" and node.name and not node.name.strip():
            skip_node = True

        if not skip_node:
            for child in child_filtered:
                child.parent = node
            node.children = child_filtered
            if child_filtered == [] and node.role == "img" or (node.name == "Image" and remove_img_if_child_img):
                # if an img node has no non-img children, remove it
                return []
            return [node]
        else:
            return child_filtered

    return dfs(root, 0, False, "", "")


def flatten_axtree(
    root: Node | list[Node],
    extra_properties: dict | None = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    filter_visible_only: bool = False,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
    ignored_properties=IGNORED_AXTREE_PROPERTIES,
    hide_bid_if_invisible: bool = False,
    hide_all_bids: bool = False,
    hide_bid_roles: list[str] = UNINTERACTIVE_ROLES,
    role_remap: dict[str, str] = ROLE_REMAP,
    hide_url: bool = True,
) -> str:
    str_list = []

    def dfs(node: Node, depth: int) -> None:
        bid = node.bid
        role = node.role
        name = node.name
        value = node.value
        indent = "\t" * depth

        attributes = []
        for property in node.properties:
            if not isinstance(property, dict):
                continue
            if "value" not in property or "value" not in property["value"]:
                continue
            prop_name = property["name"]
            prop_value = property["value"]["value"]
            if prop_name in ignored_properties:
                continue
            elif prop_name == "url" and hide_url and "Root" not in role:
                continue
            elif prop_name in ("required", "focused", "atomic"):
                if prop_value:
                    attributes.append(prop_name)
            else:
                attributes.append(f"{prop_name}={repr(prop_value)}")

        _, extra_attributes_to_print = _process_bid(
            bid,
            extra_properties=extra_properties,
            with_visible=with_visible,
            with_clickable=with_clickable,
            with_center_coords=with_center_coords,
            with_bounding_box_coords=with_bounding_box_coords,
            with_som=with_som,
            filter_visible_only=filter_visible_only,
            filter_with_bid_only=filter_with_bid_only,
            filter_som_only=filter_som_only,
            coord_decimals=coord_decimals,
        )

        attributes = extra_attributes_to_print + attributes

        if role in hide_bid_roles:
            # TODO: need list and table reformatingbid_str = ""
            bid_str = ""
        elif not (hide_all_bids or bid is None or (hide_bid_if_invisible and extra_properties is not None and extra_properties.get(bid, {}).get("visibility", 0) < 0.5)):
            bid_str = f" [{bid}]"
        else:
            bid_str = ""

        if role == "generic" and not name:
            node_str = f"{role_remap.get(role, role)}" + bid_str
        else:
            node_str = f"{role_remap.get(role, role)}{bid_str}"
            if name is not None and name.strip():
                node_str += f" {repr(name)}"

        if value is not None:
            node_str += f" value={repr(value)}"

        if attributes:
            node_str += ", ".join([""] + attributes)

        str_list.append(f"{indent}{node_str}")
        for child in node.children:
            dfs(
                child,
                depth + 1,
            )

    if isinstance(root, list):
        for node in root:
            dfs(node, 0)
    else:
        dfs(root, 0)

    return "\n".join(str_list)


def find_parent_with_bid(node: Node):
    cur = node.parent
    while cur is not None:
        if cur.bid is not None:
            return cur
        cur = cur.parent
    return None


all_action_types = ["click", "go_back", "type", "stop"]


def proper_content(content: str | None):
    if content is None:
        content = ""
    content = repr(content)[1:-1]
    return repr(content)


def click_to_code(action: str, id2node: dict[str, Node]):
    bid = action.split("[")[1].split("]")[0]
    if bid not in id2node:
        return f"Error: invalid bid [{bid}]"
    node = id2node[bid]
    if node.role == "option":
        parent = find_parent_with_bid(node)
        if parent is not None:
            return f'select_option("{parent.bid}", {proper_content(node.name)})'
    return f'click("{bid}")'


def action_to_code(action: str, id2node: dict[str, Node]):
    action = action.strip()
    action_type = action.split("[")[0].strip()

    if action_type not in all_action_types:
        return f"Error: action type {action_type} not in {all_action_types}"
    try:
        bid = action.split("[")[1].split("]")[0].strip()
    except IndexError:
        bid = None

    if action_type == "click":
        return click_to_code(action, id2node)
    elif action_type == "go_back":
        return "go_back()"
    elif action_type == "type":
        try:
            if not (action.endswith("[0]") or action.endswith("[1]")):
                # default to [0], which mean no enter after typing
                action += " [0]"
            press_enter = False if action.endswith("[0]") else True
            content = action.split("[")[2].split("]")[0]
        except Exception as e:
            return f"Error: {str(e)}"
        if press_enter:
            return f'fill("{bid}", {proper_content(content)})\npress("{bid}", "Enter")'
        else:
            return f'fill("{bid}", {proper_content(content)})'
    elif action_type == "stop":
        return f"send_msg_to_user({proper_content(bid)})"
    else:
        return f"Error: action type {action_type} not in {all_action_types}, make sure you are using the **correct action type** and **correct format**"


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


def is_valid_action(action_str: str, obs: str) -> bool:
    action_str = action_str.strip()
    try:
        action = action_str.split("[")[0].strip() if "[" in action_str else action_str.split()[0].strip()
        match action:
            case "click":
                match = re.search(r"click ?\[(\d+)\]", action_str)
                if not match:
                    return False
                element_id = match.group(1)
                if element_id in obs:
                    return True
                return False
            case "type":
                if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                    action_str += " [1]"

                match = re.search(r"type ?\[(\d+)\] ?\[(.*)\] ?\[(\d+)\]", action_str, re.DOTALL)
                if not match:
                    return False
                element_id, _, enter_flag = (
                    match.group(1),
                    match.group(2),
                    match.group(3) == " " or match.group(3) == "",
                )
                assert enter_flag == "0" or enter_flag == "1"
                if element_id in obs:
                    return True
            case "go_back":
                return True
            case "go_home":
                return True
            case "note":
                return True
            case "stop":
                return True
        return False
    except Exception as e:
        print(f"Error checking action validity: {e}")
        return False


class NotePad:
    def __init__(self):
        self.notes = []
        self.last_note = None

    def append_note(self, note: str, step=None):
        self.last_note = note
        if step is not None:
            note = f"Step {step}: {note}"
        self.notes.append(note)

    def is_empty(self) -> bool:
        return len(self.notes) == 0

    def is_repeating(self, note: str) -> bool:
        return note == self.last_note

    def get_notes(self):
        return "\n".join(self.notes)

    def get_last(self):
        return self.notes[-1]


SYSTEM_PROMPT = """You are an AI assistant performing tasks on a web browser.

You will be provided with:
- OBJECTIVE: the task you must complete.
- MEMORY SUMMARY: an append-only compact summary of ALL previous steps so far (from step 1 to the latest), maintained for you. Treat this as the authoritative history and use it to stay consistent and avoid repeating actions.
- CURRENT OBSERVATION: the current page’s accessibility text relevant for deciding the next action.

Your job is to think, then issue exactly ONE action for the current step based ONLY on the OBJECTIVE, the MEMORY SUMMARY, and the CURRENT OBSERVATION.

Decision policy (very important):
- Minimize steps: prefer the fastest, most direct action that advances the OBJECTIVE.
- Prioritize high-yield interactions (e.g., clear navigation buttons, search fields, filters) over low-value actions (aimless repeated clicks, irrelevant elements).
- Never repeat an action if the page has not changed.
- As soon as the CURRENT OBSERVATION indicates the OBJECTIVE is achieved, immediately stop.
- If a dropdown/menu/dialog is already open/expanded, click the relevant target item (e.g., "New project") instead of clicking the opener again.

Allowed actions (exact formats only):
- click [id]
- type [id] [content] [press_enter_after=0|1]
- stop [answer]
- go_back
Only output one ACTION per step, and note that if you believe the OBJECTIVE has been completed, output ONLY `stop [answer]` and nothing else.

Output format (use these section headers exactly and do NOT add extra lines beyond them):
OBSERVATION DESCRIPTION:
Briefly describe ONLY the CURRENT OBSERVATION and any key UI that matters.
REASON:
Concise rationale for the next action.
ACTION:
Exactly ONE line in bracket format ONLY (e.g., `click [7]`, `type [15] [query] [1]`, `go_back`, or `stop [answer]`). **CRITICAL**: Never omit the ACTION header; always include `ACTION:` followed by exactly ONE bracket-style action line. If you believe the objective is achieved, output ONLY `stop [answer]` as the ACTION and nothing else.
OBSERVATION HIGHLIGHT:
List element ids on the current page that justify your action, sorted by relevance (e.g., `1321, 52, 756`).
"""


SUMMARY_SYSTEM_PROMPT = """You are the memory summarizer for a web browsing agent.
You update the rolling memory using ONLY the previous step information:
- the previous memory text (may be empty),
- the previous observation (page text),
- the previous final action string.

Strict output contract (append-only):
- If the PREVIOUS MEMORY shows "None" (meaning this is the very first summary), you MUST start with exactly ONE line: `Goal: <objective>` where <objective> is copied from the OBJECTIVE field provided.
- Then return EXACTLY THREE lines for the previous step:
  `Step <k>:`
  `page: <precise, non-generic page state + several concise, goal-helpful details>`
  `action: <previous final action string and concise description, e.g. action: click [6002], click on the "Create project" button>`
- The `page:` line MUST be accurate and non-generic and use several short sentences: first, state exactly where we are / what the page is (e.g., current view/section/state). Then add goal-relevant details that help progress. You may (a) mention specific UI items only when critical to the objective, or (b) give compact grouped descriptions (e.g., "buttons related to checkout", "filters for price/brand", "form fields to enter shipping info"). Prefer information that advances the objective; avoid element ids, CSS, and irrelevant clutter; keep it concise.
- Do NOT rewrite or repeat earlier memory. Do NOT include any earlier lines. Return ONLY the new lines to append (the update), no extra headings, comments, quotes, or blank pre/post padding.

CRITICAL NUMBERING RULES (STRICT):
- Compute k as follows:
  * If PREVIOUS MEMORY is "None", then k = 1 (this is the first step)
  * Otherwise, look at the last "Step X:" line in PREVIOUS MEMORY and set k = X + 1
- NEVER skip or invent numbers. Follow this rule exactly.
- Example: if PREVIOUS MEMORY is "None", output "Goal: <objective>\nStep 1:\n..."
- Example: if PREVIOUS MEMORY ends with "Step 3:", output "Step 4:\n..."
"""



class WebArenaAgent(BaseAgent):
    def __init__(self):
        self.mode = "Step"
        self.use_html = False
        self.use_axtree = True
        self.use_screenshot = False
        self.use_note = False

        assert self.use_note == False, "Note is not supported for now"

        self.action_history = []  # all are in string
        self.id2node = {}
        self.past_obs = []
        self.interaction_histories = []
        self.interaction_history_str = ""
        self.objective = None
        self.notepad = NotePad()
        # Compact rolling memory summary maintained by the agent
        self.summary: str | None = None
        # Keep the most recent summary prompt (system+user) for logging placement
        self._last_summary_prompt_messages: list[dict[str, str]] | None = None

        # for interface compliance
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0
        self.reset()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.
        Includes logic to check if the observation changed from the previous step.
        """
        obs = self._preproc_obs(observation)

        # Update the last step in the trajectory with the outcome (next_observation, reward, done, info)
        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            # The observation received here is the 'next_observation' for the *previous* action/step
            # Store the next observation in the info dict since Step doesn't have a next_observation field
            prior_step.info["next_observation"] = obs["axtree_txt"]
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info.update(info)

        # Only append a new [system, user] pair if the episode is not done
        if not done:
            user_prompt = self._get_user_prompt(obs)
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            self.messages.extend(message)

        # Create a new step for the current state (with the observation that resulted from the last action)
        # This step's action, reward, etc., will be filled in by subsequent update_from_model and update_from_env calls
        if done:
            return

        cur_step = Step(observation=obs["axtree_txt"])
        self._trajectory.steps.append(cur_step)
        assert "axtree_txt" in obs, "Only axtree is supported for now"
        self.past_obs.append(obs["axtree_txt"])

    def update_from_model(self, response: str, **kwargs):
        content = response
        
        # DEBUG: Verify this response belongs to this trajectory
        import logging
        trajectory_id = getattr(self, '_trajectory_id', 'unknown')
        logging.debug(f"[{trajectory_id}] update_from_model called, trajectory has {len(self._trajectory.steps)} steps")

        assert self._trajectory.steps, "Trajectory should not be empty when update_from_model is called."

        # Parse structured sections, split out UPDATED MEMORY SUMMARY so that
        # assistant content only keeps REASON/ACTION/etc. (summary will be added as a tool message)
        result, action_code, updated_summary = self._parse_model_response(content)
        # Reconstruct assistant-visible content without the UPDATED MEMORY SUMMARY section
        # Extract a single clean bracket-action line for display and for safety
        clean_action = ""
        try:
            _m = re.search(
                r"(click\s*\[\d+\]|type\s*\[\d+\]\s*\[.*?\]\s*\[(?:0|1)\]|go_back|stop\s*\[.*?\])",
                result.get("ACTION", ""),
                flags=re.IGNORECASE | re.DOTALL,
            )
            if _m:
                clean_action = _m.group(1).strip()
        except Exception:
            clean_action = result.get("ACTION", "").strip()

        # 按你的要求：action assistant 直接保留模型原文，不做重排/裁剪
        self.messages.append({"role": "assistant", "content": content})
        # print("Parsed result: ", result, "action_code: ", action_code)

        cur_step = self._trajectory.steps[-1]
        cur_step.thought = result.get("REASON", "")
        cur_step.action = action_code
        cur_step.model_response = content

        # Save a compact interaction record with the raw bracket action for memory enrichment
        self.interaction_histories.append(
            {
                "obs_short": result.get("OBSERVATION DESCRIPTION", ""),
                "reason_for_action": result.get("REASON", ""),
                # Store the raw bracket-style action if available to help summary enrichment
                "action": clean_action or result.get("ACTION", ""),
            }
        )
        # Keep a history of bracket-style actions for dedup and analysis
        try:
            if clean_action:
                self.action_history.append(clean_action)
        except Exception:
            pass

        self.step += 1
        return Action(action=action_code)

    def _get_interaction_history_str(self, cur_obs):
        interaction_history_str = ""
        assert len(self.past_obs) == self.step
        for idx in range(self.step):
            obs_short = self.interaction_histories[idx]["obs_short"]
            reason_for_action = self.interaction_histories[idx]["reason_for_action"]
            action_str = self.interaction_histories[idx]["action"]
            if self.past_obs[idx] == cur_obs:
                obs_short = "The same as the CURRENT OBSERVATION (see below CURRENT OBSERVATION section)."
            interaction_history_str += f"<step_{idx}_interaction>\nOBSERVATION:\n{obs_short}\nREASON FOR ACTION:{reason_for_action}\nACTION:\n{action_str}\n</step_{idx}_interaction>\n"

        return interaction_history_str

    @property
    def system_prompt(self):
        return SYSTEM_PROMPT

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self.messages

    def get_prompt(self):
        # Return the most recent ACTION prompt pair: [system(self.system_prompt), user]
        # Do NOT return summary prompts (whose system content != self.system_prompt).
        assert self.messages, "Invariant violation: no messages available before generation"
        sys_idx = -1
        user_idx = -1
        for i in range(len(self.messages) - 2, -1, -1):
            if self.messages[i].get("role") == "system" and self.messages[i].get("content") == self.system_prompt:
                if i + 1 < len(self.messages) and self.messages[i + 1].get("role") == "user":
                    sys_idx = i
                    user_idx = i + 1
                    break
        assert sys_idx != -1 and user_idx != -1, "Invariant violation: no ACTION prompt pair found before generation"
        return [self.messages[sys_idx], self.messages[user_idx]]

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self):
        self._trajectory = Trajectory()
        self.past_obs = []
        self.interaction_histories = []
        self.interaction_history_str = ""
        self.messages = []
        self.notepad = NotePad()
        self.summary = None
        self.step = 0
        self.objective = None
        self.action_history = []  # all are in string
        self.id2node = {}

    def get_current_state(self) -> Step:
        if not self._trajectory.steps:
            raise ValueError("get_current_state called before the first observation was processed.")
        return self._trajectory.steps[-1]

    def _get_user_prompt(self, obs):
        assert self.objective is not None
        assert self.use_axtree, "Only axtree is supported for now"

        summary = (self.summary or "None") if self.step > 0 else "None"
        # Clip observation text to avoid exceeding model context on rollout server side
        axt = obs["axtree_txt"]
        try:
            import os
            max_chars = int(os.environ.get("RLLM_AXTREE_MAX_CHARS", "15000"))
        except Exception:
            max_chars = 15000
        if isinstance(axt, str) and len(axt) > max_chars:
            axt = axt[:max_chars]
        user_prompt_template = """OBJECTIVE:
{objective}

MEMORY SUMMARY:
{summary}

CURRENT OBSERVATION:
{observation}
"""
        user_prompt = user_prompt_template.format(
            objective=self.objective or "",
            summary=summary,
            observation=axt,
        )
        return user_prompt

    def get_summary_prompt(self) -> list[dict[str, str]] | None:
        """Build a prompt to update memory for the step that just produced ACTION.

        Called immediately AFTER the agent outputs ACTION for the current step.
        Summarizes the current last step (o_t, a_t) with step number k = len(steps).
        Returns None for the very first call if no step exists.
        """
        steps = self._trajectory.steps
        if len(steps) == 0:
            return None
        prev_step = steps[-1]
        prev_obs = prev_step.observation if isinstance(prev_step.observation, str) else ""
        # Clip previous observation as well to keep summary prompt short
        try:
            import os
            max_chars = int(os.environ.get("RLLM_AXTREE_MAX_CHARS", "15000"))
        except Exception:
            max_chars = 15000
        if isinstance(prev_obs, str) and len(prev_obs) > max_chars:
            prev_obs = prev_obs[:max_chars]
        # Prefer the last recorded bracket-style action for readability
        prev_action = ""
        try:
            if self.interaction_histories:
                prev_action = self.interaction_histories[-1].get("action", "")
        except Exception:
            prev_action = ""
        prev_summary = self.summary or ""
        objective = self.objective or ""

        user_prompt = (
            f"OBJECTIVE:\n{objective}\n\n"
            f"PREVIOUS MEMORY:\n{prev_summary if prev_summary else 'None'}\n\n"
            f"PREVIOUS OBSERVATION:\n{prev_obs}\n\n"
            f"PREVIOUS ACTION:\n{prev_action}"
        )
        msgs = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # Cache for later insertion into the conversation log, preserving order
        self._last_summary_prompt_messages = msgs
        return msgs

    def update_summary_from_model(self, response: str):
        """Parse summary-only response and update rolling memory.

        Also append as a tool message to the chat (masked in training by default).
        """
        raw = (response or "").strip()
        if not raw:
            return
        
        # DEBUG: Verify this summary belongs to this trajectory
        import logging
        trajectory_id = getattr(self, '_trajectory_id', 'unknown')
        logging.debug(f"[{trajectory_id}] update_summary_from_model called, trajectory has {len(self._trajectory.steps)} steps")

        # Compute expected k locally (number of completed actions)
        expected_k = len(self._trajectory.steps)
        
        # CRITICAL: Verify expected_k matches current trajectory state
        current_steps = len(self._trajectory.steps)
        if expected_k is not None and expected_k > current_steps:
            logging.error(f"[{trajectory_id}] SUMMARY MISMATCH! expected_k={expected_k} > current_steps={current_steps}. "
                         f"This indicates summary from step {expected_k} is being applied to a trajectory that's only at step {current_steps}. "
                         f"Skipping to prevent cross-trajectory contamination.")
            return

        # 不再对 summary 文本做语义重写/规范化，保留原文；仅做顺序校验

        # Avoid duplicating the same Step k in memory
        try:
            import re as _re
            if self.summary and _re.search(rf"(?m)^Step\s+{expected_k}\s*:\s*$", self.summary):
                # Already summarized; do not insert again
                return
        except Exception:
            pass

        # Append-only update of rolling memory（使用模型原文）
        if self.summary:
            if not self.summary.endswith("\n"):
                self.summary += "\n"
            self.summary = self.summary + raw
        else:
            self.summary = raw

        # CRITICAL: Simply APPEND summary messages at the end
        # The engine ensures we await pending_summary_task BEFORE the next step begins,
        # so messages order is: [sys0, user0, asst0(action)] → append summary → [sys1, user1] → append asst1
        trajectory_id = getattr(self, '_trajectory_id', 'unknown')
        
        import logging
        logging.debug(f"[{trajectory_id}] Appending summary to messages (current length: {len(self.messages)})")
        
        # Append summary prompt messages (system, user)
        if self._last_summary_prompt_messages and isinstance(self._last_summary_prompt_messages, list):
            for m in self._last_summary_prompt_messages:
                self.messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        
        # Append summary assistant response（保留模型原文）
        self.messages.append({"role": "assistant", "content": raw})

    def refresh_action_prompt(self):
        """Refresh the current step's user prompt to reflect the latest summary.

        Assumes the last two messages are [system, user] added by update_from_env.
        """
        assert self.messages, "Invariant violation: no messages available when refreshing action prompt"
        cur_obs = ""
        state = self.get_current_state()
        if state and isinstance(state.observation, str):
            cur_obs = state.observation
        obs = {"axtree_txt": cur_obs}
        new_user_prompt = self._get_user_prompt(obs)
        # Prefer updating the user message that follows the last ACTION system prompt
        # to avoid overwriting summary prompts.
        target_user_idx = -1
        for i in range(len(self.messages) - 2, -1, -1):
            if self.messages[i].get("role") == "system" and self.messages[i].get("content") == self.system_prompt:
                if i + 1 < len(self.messages) and self.messages[i + 1].get("role") == "user":
                    target_user_idx = i + 1
                    break
        if target_user_idx == -1:
            # Fallback to the most recent user message
            for i in range(len(self.messages) - 1, -1, -1):
                if self.messages[i].get("role") == "user":
                    target_user_idx = i
                    break
        assert target_user_idx != -1, "Invariant violation: no user message found when refreshing action prompt"
        self.messages[target_user_idx]["content"] = new_user_prompt

    def _preproc_obs(self, obs: dict) -> dict:
        if "goal_object" in obs:
            self.objective = obs["goal_object"][0]["text"]
        result = from_axtree_construct_tree(obs["axtree_object"], return_id2node=True)
        if isinstance(result, tuple):
            root, id2node = result
        else:
            # This shouldn't happen since we pass return_id2node=True, but handle it for type safety
            root = result
            id2node = {}

        root = clean_accesibility_tree(root)
        roots = prune_axtree(root)
        self.id2node = id2node
        return {
            "chat_messages": obs["chat_messages"],
            # "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            "axtree_txt": flatten_axtree(roots, hide_url=False),
            # "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }

    def _parse_model_response(self, response):
        """
        Robustly parse structured sections (case/colon tolerant). If ACTION block is
        missing, fall back to extracting the last bracket-style action from the full text.

        Returns:
            Tuple[dict[str, str], str, str]: (parts_dict, action_code, updated_summary)
        """
        canonical_parts = [
            "INTERACTION HISTORY SUMMARY",
            "OBSERVATION DESCRIPTION",
            "REASON",
            "ACTION",
            "UPDATED MEMORY SUMMARY",
            "OBSERVATION HIGHLIGHT",
        ]
        aliases = {
            "INTERACTION HISTORY SUMMARY": ["INTERACTION HISTORY SUMMARY", "INTERACTION HISTORY", "HISTORY SUMMARY"],
            "OBSERVATION DESCRIPTION": ["OBSERVATION DESCRIPTION", "OBSERVATION", "OBSERVATION SUMMARY", "OBSERVATION DESC"],
            "REASON": ["REASON", "REASONING", "THOUGHT", "THINKING"],
            "ACTION": ["ACTION", "ACTIONS"],
            "UPDATED MEMORY SUMMARY": ["UPDATED MEMORY SUMMARY", "MEMORY SUMMARY", "SUMMARY UPDATE", "MEMORY UPDATE"],
            "OBSERVATION HIGHLIGHT": ["OBSERVATION HIGHLIGHT", "OBSERVATION HIGHLIGHTS", "HIGHLIGHTS", "KEY ELEMENTS"],
        }
        alias_to_canonical = {}
        for k, vs in aliases.items():
            for v in vs:
                alias_to_canonical[v.upper()] = k

        result: dict[str, Any] = {}
        current_part: str | None = None

        for raw_line in response.split("\n"):
            line = raw_line.rstrip()
            header = line.strip().rstrip(":").strip().upper()
            if header in alias_to_canonical:
                current_part = alias_to_canonical[header]
                if current_part not in result:
                    result[current_part] = []
                continue
            if current_part:
                result.setdefault(current_part, []).append(line.strip())

        for key in list(result.keys()):
            if isinstance(result[key], list):
                result[key] = "\n".join(result[key]).strip()

        action_str = (result.get("ACTION", "") or "").strip()
        if not action_str:
            # Fallback: search the entire response for a bracket-style action (prefer last)
            try:
                pattern = re.compile(
                    r"(click\s*\[\d+\]"
                    r"|type\s*\[\d+\]\s*\[.*?\]\s*\[(?:0|1)\]"
                    r"|go_back"
                    r"|stop\s*\[.*?\])",
                    flags=re.IGNORECASE | re.DOTALL,
                )
                matches = list(pattern.finditer(response))
                if matches:
                    action_str = matches[-1].group(1).strip()
                    # Also write back into parts so assistant content can show ACTION
                    result["ACTION"] = action_str
            except Exception:
                action_str = ""

        action_code = action_to_code(action_str, self.id2node)
        updated_summary = (result.get("UPDATED MEMORY SUMMARY", "") or "").strip()
        return result, action_code, updated_summary

    def compute_training_reward(self, trajectory: Trajectory) -> float:
        if not trajectory:
            return 0
        # print(trajectory.steps)
        reward = trajectory.steps[-1].reward
        reward_penalty = 0
        # for step in trajectory.steps:
        #     if not self.validate_step(step):
        #         reward_penalty = -0.5
        #         break
        return reward + reward_penalty
