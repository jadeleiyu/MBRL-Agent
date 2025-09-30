import os
import json
import argparse

os.environ["HF_HUB_CACHE"] = "/checkpoint/multimodal-reasoning/jadeleiyu/huggingface"
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"   # set this *before* importing vllm
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"  # or "ERROR"
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer
import datasets
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
from tqdm import tqdm

"""
python wm_cot_gen.py --job_id 0 
"""

# we first need to generate CoT for the world model so that it could predict the next acc-tree via reasoning

SYS_PROMPT_WM_COT_GEN = """You are an intelligent agent that predicts next state from given current action in a web environment, with your own logical reasoning. 

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The previous action: This is the action you just performed in the previous step. It may be helpful to track your progress. 
The current action: This is the current action that you performed to achieve the user's objective in the current web page's accessibility tree.
The format of previous actions can fall into several categories:
Page Operation Actions:

```click [id]```: This action clicks on an element with a specific id on the webpage.
```type [id] [content]```: Use this to type the content into the field with id. By default, the 'Enter' key is pressed after typing unless press_enter_after is set to 0, i.e., ```type [id] [content] [0]```.
```hover [id]```: Hover over an element with id.
```press [key_comb]```: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
```scroll [down]``` or ```scroll [up]```: Scroll the page up or down.

Tab Management Actions:
```new_tab```: Open a new, empty browser tab.
```tab_focus [tab_index]```: Switch the browser's focus to a specific tab using its index.
```close_tab```: Close the currently active tab.

URL Navigation Actions:
```goto [url]```: Navigate to a specific URL.
```go_back```: Navigate to the previously viewed page.
```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed)

Completion Action:
```stop [answer]```: Done when you believe the task is complete.

You will also be given the ground-truth next web page's accessibility tree.
Your task is to generate descriptions of the changes to the current web page observation by the action that result in the given next web page accessibility tree, as well as the reasoning process of how you come up with the changes.
Follow the rules below when generating the web state changes and reasoning traces:

1. When you generate the state change description, you must identify and mention only the changed parts of the accessibility tree for the next state based on the given current action. 
2. Your generated state change description should be informative, so that another language model could accurately predict the next web page tree based on the current page and your generated state changes. 
3. Generate your answer with the following format: 
[Rationale]\n\{rationale\}\n\n\n[Web state changes]\n\{changes\} 
where \{rationale\} is your reasoning trace, and \{changes\} is the web state change description.
"""

USER_PROMPT_WM_COT_GEN = """User objective: {usr_obj}
Current web page accessibility tree: {curr_acc_tree}
Current web url: {curr_url}
Previous action: {prev_action}
Current action: {curr_action}
Next web page accessibility tree: {next_acc_tree}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--input_dir", type=str, default="/home/jadeleiyu/projects/mbrl_agent/world_model/sft_data/")
    parser.add_argument("--output_dir", type=str, default="/home/jadeleiyu/projects/mbrl_agent/world_model/sft_data/")
    
    args = parser.parse_args()

    output_fn = os.path.join(args.output_dir, f"wm_cot_gen_outputs_{args.job_id}.json")
    if not os.path.isfile(output_fn):
    
        # --- 1) Render the prefill with Harmony ---
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        model = LLM(
            model=args.model_name,
            trust_remote_code=True,
            tensor_parallel_size=8,
            dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.9,
            max_tokens=4096,
        )

        with open(os.path.join(args.input_dir, "wm_cot_gen_inputs.json"), 'r') as f:
            wm_cot_ds = json.load(f)

        job_chunk_size = int(len(wm_cot_ds) / args.n_jobs)
        start = args.job_id * job_chunk_size
        end = min(len(wm_cot_ds), start + job_chunk_size)
        wm_cot_ds = wm_cot_ds[start:end]

        stop_token_ids = encoding.stop_tokens_for_assistant_actions()
        sampling_params.stop_token_ids=stop_token_ids

        n_batch = int(len(wm_cot_ds) / args.batch_size) + 1
        for i in tqdm(range(n_batch)):
        # for i in tqdm(range(2)):
            batch_start = i * args.batch_size
            batch_end = min(len(wm_cot_ds), batch_start + args.batch_size)
            batch_input_ids = []
            for j in range(batch_start, batch_end):
                example = wm_cot_ds[j]
                user_input = USER_PROMPT_WM_COT_GEN.format(
                    usr_obj=example['objective'],
                    curr_acc_tree=example['current_observation'],
                    curr_url=example['current_url'],
                    prev_action=example['previous_action'],
                    curr_action=example['current_action'],
                    next_acc_tree=example['next_observation'],
                )
                convo = Conversation.from_messages(
                    [
                        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                        Message.from_role_and_content(
                            Role.DEVELOPER,
                            DeveloperContent.new().with_instructions(SYS_PROMPT_WM_COT_GEN),
                        ),
                        Message.from_role_and_content(Role.USER, user_input),
                    ]
                )
                batch_input_ids.append(encoding.render_conversation_for_completion(convo, Role.ASSISTANT))

            outputs = model.generate(
                prompt_token_ids=batch_input_ids,  
                sampling_params=sampling_params,
                use_tqdm=False
            )
            for k in range(len(batch_input_ids)):
                output_tokens = outputs[k].outputs[0].token_ids
                entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
                wm_cot_ds[batch_start+k]['wm_cot'] = entries[1].to_dict()['content'][0]['text']
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_fn = os.path.join(args.output_dir, f"wm_cot_gen_outputs_{args.job_id}.json")
        with open(output_fn, 'w') as f:
            json.dump(wm_cot_ds, f)


if __name__ == "__main__":
    main()