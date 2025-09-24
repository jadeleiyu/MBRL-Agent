import os
import json
import argparse

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"   # set this *before* importing vllm
os.environ["VLLM_LOGGING_LEVEL"]    = "WARNING"  # or "ERROR"
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer
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
python wm_live_cot_gen.py --start 0 --end 3000
python wm_live_cot_gen.py --start 3000 --end 5000
"""

nnet_live_dir = '/home/jadeleiyu/projects/mbrl_agent/world_model/sft_data/Nnetnav/live/'

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
Your task is to generate a logical reasoning rationale that explains how you can predict the next web page accessibility tree given the provided information.
Follow the following rules for generating the reasoning rationale.
1. Please generate your answer starting with Let's think step by step, with your logical REASONING.
2. When you generate your logical reasoning, you must identify and mention only the changed parts of the [accessibility tree] for the next state based on the given current action. 
3. Generate your answer with a starting phrase "[Rationale]\n".
"""

USER_PROMPT_WM_COT_GEN = """User objective: {usr_obj}
Current web page accessibility tree: {curr_acc_tree}
Previous action: {prev_action}
Current action: {curr_action}
Next web page accessibility tree: {next_acc_tree}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=5000)
    args = parser.parse_args()
    
    # --- 1) Render the prefill with Harmony ---
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    model = LLM(
        model="openai/gpt-oss-20b",
        trust_remote_code=True,
        tensor_parallel_size=8,
        dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        max_tokens=4096,
    )

    with open(os.path.join(nnet_live_dir, "train_wm.jsonl"), 'r') as f:
        nnet_live_wm_ds = json.load(f)

    nnet_live_wm_ds = list(nnet_live_wm_ds.items())[args.start:args.end]

    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    sampling_params.stop_token_ids=stop_token_ids

    for i in tqdm(range(len(nnet_live_wm_ds))):
    # for i in tqdm(range(5)):
        try:
            example = nnet_live_wm_ds[i]
            input_token_ids = []
            for step in example[1]['steps']:
                # messages = [
                #     {'role': 'system', 'content': SYS_PROMPT_WM_COT_GEN},
                #     {'role': 'user', 'content': }
                # ]
                user_input = USER_PROMPT_WM_COT_GEN.format(
                    usr_obj=example[1]['objective'],
                    curr_acc_tree=step['current_observation'],
                    # curr_url=step['url'],
                    prev_action=step['previous_actions'].split('\n')[-1],
                    curr_action=step['current_action'],
                    next_acc_tree=step['next_observation'],
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
                input_token_ids.append(encoding.render_conversation_for_completion(convo, Role.ASSISTANT))
            
            outputs = model.generate(
                prompt_token_ids=input_token_ids,  
                sampling_params=sampling_params,
                use_tqdm=False
            )
            for j in range(len(example[1]['steps'])):
                try:
                    output_tokens = outputs[j].outputs[0].token_ids
                    entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
                    # for message in entries:
                    #     print(f"{json.dumps(message.to_dict())}")
                    # print('\n\n\n')
                    try:
                        wm_cot = entries[1].to_dict()['content'][0]['text'].split("[Rationale]\n")[1]
                    except Exception as e:
                        try:
                            wm_cot = entries[1].to_dict()['content'][0]['text']
                        except Exception as e:
                            wm_cot = 'None'
                    nnet_live_wm_ds[i][1]['steps'][j]['wm_cot'] = wm_cot
                except Exception as e:
                    nnet_live_wm_ds[i][1]['steps'][j]['wm_cot'] = "None"

        except Exception as e:
            pass


    with open(f'/home/jadeleiyu/projects/mbrl_agent/world_model/sft_data/Nnetnav/live/train_wm_with_cot_{args.start}_{args.end}.jsonl', 'w') as f:
        json.dump(nnet_live_wm_ds, f)


if __name__ == "__main__":
    main()