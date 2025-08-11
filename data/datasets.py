from datasets import load_dataset
import json
import lxml

from dom_utils import get_tree_repr
from ..utils.prompting import WEBRL_SYSTEM_PROMPT, MIND2WEB_SYSTEM_PROMPT


def get_gt_element(sample, dom_tree, gt_node_id, id_mapping):
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    node = candidate_nodes[id_mapping.get(gt_node_id, -1)]
    element = [
        node.attrib["backend_node_id"],
        " ".join(
            get_tree_repr(
                node,
                id_mapping=id_mapping,
                keep_html_brackets=sample.get("keep_html_brackets", False),
            )[0].split()[:10]
        )
    ]
    return element


def prepare_sft_data(webrl_path, mind2web_path):
    sft_ds = []

    ##### WebRL SFT data #####
    if webrl_path:
        data = _load_local_json_like(webrl_path)
        for ex in data:
            messages = {
                'messages':[
                    {"role": "system", "content": WEBRL_SYSTEM_PROMPT},
                    {"role": "user", "content": ex['conversation'][0]['value']},
                    {"role": "assistant", "content": ex['conversation'][1]['value']},
                ]
            }
            sft_ds.append(messages)

    ##### Mind2Web SFT data #####
    if mind2web_path:
        data = load_dataset("osunlp/Mind2Web", split='train')
        for ex in data:
            n_action_steps = len(ex.get('actions', []))
            if n_action_steps == 0:
                continue

            prev_actions = ""
            dom_tree = lxml.etree.fromstring(ex["cleaned_html"])
            tree_repr, id_mapping = get_tree_repr(
                dom_tree, id_mapping={}
            )
            for i in range(n_action_steps):
                pos_candidates_i = ex['actions'][i]['pos_candidates']
                if pos_candidates_i:
                    pos_candidate_i = pos_candidates_i[0]
                    gt_node_id = pos_candidate_i['backend_node_id']
                    element = get_gt_element(ex, dom_tree, gt_node_id, id_mapping)

                    instruction = (
                        f"Task: {ex['confirmed_task']}\n"
                        f"Website: {ex.get('website', '')} | "
                        f"Domain: {ex.get('domain', '')} | "
                        f"Subdomain: {ex.get('subdomain', '')} | "
                        f"HTML of the starting webpage: {tree_repr}"
                        f"Previous actions: \n{prev_actions}\n\n\n"
                        "What should be the next action?"
                        "Please generate the element to interact with, the action to perform, and the value to type in or select. "
                        "If the task cannot be completed, output None."
                    )

                    action_i = (
                        f"Element: {element}; "
                        f"Operation: {ex['actions'][i]['operation']['op']}; "
                        f"Value: {ex['actions'][i]['operation']['value']}."
                    )
                    messages = {
                        'messages':[
                            {"role": "system", "content": MIND2WEB_SYSTEM_PROMPT},
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": action_i},
                        ]
                    }
                    sft_ds.append(messages)
                    prev_actions += f"Action step {i}: {action_i}\n"

    return sft_ds


def _load_local_json_like(path: str):
    """Best-effort loader for JSON / JSONL / {"data": [...]} files."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        # try dict with 'data'
        try:
            obj = json.load(f)
            if isinstance(obj, dict) and "data" in obj:
                return obj["data"]
        except Exception:
            pass
        f.seek(0)
        # fallback jsonl
        return [json.loads(line) for line in f if line.strip()]

