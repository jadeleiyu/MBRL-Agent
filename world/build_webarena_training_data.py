#!/usr/bin/env python3
"""
构建类似run_webarena_world_model.sh中使用的训练数据
基于world文件夹中的方法和数据格式
"""

import json
import os
import argparse
from typing import List, Dict, Any
from datasets import load_dataset
import pandas as pd


def parse_nnetnav_task(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    从stanfordnlp/nnetnav数据集中解析任务信息
    类似于collect_dream_trajs.py中的解析逻辑
    """
    text = sample["messages"][1]["content"]
    
    # 解析instruction (OBJECTIVE)
    instruction = text.split('\nOBJECTIVE:')[1].split('\nPREVIOUS ACTIONS')[0].strip()
    
    # 解析初始accessibility tree (OBSERVATION)
    acc_tree = text.split('OBSERVATION:\n')[1].split('\nURL:')[0].strip()
    
    # 解析URL
    url = text.split('\nURL:')[1].splitlines()[0].strip() if '\nURL:' in text else ''
    
    return {
        "instruction": instruction,
        "acc_tree": acc_tree,
        "url": url
    }


def build_webarena_prompt(instruction: str, acc_tree: str, url: str = "") -> List[Dict[str, str]]:
    """
    构建WebArena风格的prompt，用于训练policy model
    基于rllm/agents/webarena_agent.py中的格式
    """
    system_prompt = """You are an AI assistant performing tasks on a web browser. You will be provided with task objective, current step, web page observations, interaction history and previous taked notes. You need to issue an action for this step.

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

    user_prompt = f"""OBJECTIVE:
{instruction}
CURRENT OBSERVATION:
{acc_tree}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def create_webarena_training_data(env: str = "live", 
                                 output_file: str = "webarena_training_data.parquet",
                                 max_samples: int = 1000) -> None:
    """
    创建WebArena风格的训练数据，用于PPO训练
    
    Args:
        env: 环境名称 (live, wa等)
        output_file: 输出文件路径
        max_samples: 最大样本数
    """
    print(f"加载数据集: stanfordnlp/nnetnav-{env}")
    ds = load_dataset(f"stanfordnlp/nnetnav-{env}", split='train')
    
    training_data = []
    
    for i, sample in enumerate(ds):
        if i >= max_samples:
            break
        try:
            task_info = parse_nnetnav_task(sample)
            instruction = task_info["instruction"]
            acc_tree = task_info["acc_tree"]
            url = task_info["url"]

            messages = build_webarena_prompt(instruction, acc_tree, url)
            
            data_entry = {
                "data_source": f"stanfordnlp/nnetnav-{env}",
                "prompt": messages,
                "ability": "web",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "index": i,
                    "instruction": instruction,
                    "url": url,
                    "acc_tree": acc_tree,
                    "task": {
                        "objective": instruction,
                        "axtree_txt": acc_tree,
                        "url": url
                    }
                },
                "task": {
                    "objective": instruction,
                    "axtree_txt": acc_tree,
                    "url": url
                }
            }
            
            training_data.append(data_entry)
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1} 个样本")
                
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            continue
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # ===== ✅ 保存为 Parquet 文件 =====
    df = pd.DataFrame(training_data)
    df.to_parquet(output_file, index=False)
    print(f"✅ 成功创建 {len(training_data)} 个训练样本")
    print(f"🟢 Parquet 数据保存到: {output_file}")
    
    # ===== ✅ 同时保存为 JSONL 文件 =====
    jsonl_output = os.path.splitext(output_file)[0] + ".jsonl"
    with open(jsonl_output, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"🟢 JSONL 数据保存到: {jsonl_output}")

    # 显示示例
    if training_data:
        print("\n示例数据:")
        example = training_data[0]
        print(f"Instruction: {example['extra_info']['instruction']}")
        print(f"URL: {example['extra_info']['url']}")
        print(f"Prompt messages: {len(example['prompt'])} 条")
        if example['prompt']:
            print(f"第一条消息长度: {len(example['prompt'][0]['content'])} 字符")
            print(f"第一条消息预览: {example['prompt'][0]['content'][:200]}...")



def create_simple_instruction_data(env: str = "live", 
                                 output_file: str = "simple_instructions.jsonl",
                                 max_samples: int = 1000) -> None:
    """
    创建简单的instruction数据，用于快速测试
    """
    print(f"加载数据集: stanfordnlp/nnetnav-{env}")
    ds = load_dataset(f"stanfordnlp/nnetnav-{env}", split='train')
    
    instructions = []
    
    for i, sample in enumerate(ds):
        if i >= max_samples:
            break
            
        try:
            # 解析任务信息
            task_info = parse_nnetnav_task(sample)
            instruction = task_info["instruction"]
            acc_tree = task_info["acc_tree"]
            url = task_info["url"]
            
            # 构建简单的instruction数据
            data_entry = {
                "instruction": instruction,
                "initial_observation": acc_tree,
                "url": url,
                "index": i
            }
            
            instructions.append(data_entry)
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1} 个样本")
                
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            continue
    
    # 保存数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in instructions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"成功创建 {len(instructions)} 个instruction样本")
    print(f"数据保存到: {output_file}")
    
    # 显示示例
    if instructions:
        print("\n示例instruction:")
        example = instructions[0]
        print(f"Instruction: {example['instruction']}")
        print(f"URL: {example['url']}")
        print(f"Initial observation长度: {len(example['initial_observation'])} 字符")


def main():
    parser = argparse.ArgumentParser(description="构建WebArena训练数据")
    parser.add_argument("--env", type=str, default="live", 
                       help="环境名称 (live, wa等)")
    parser.add_argument("--output", type=str, default="webarena_training_data.parquet",
                       help="输出文件路径")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="最大样本数")
    parser.add_argument("--mode", type=str, default="webarena",
                       choices=["webarena", "simple"],
                       help="数据格式模式")
    
    args = parser.parse_args()
    
    if args.mode == "webarena":
        create_webarena_training_data(
            env=args.env,
            output_file=args.output,
            max_samples=args.max_samples
        )
    else:
        create_simple_instruction_data(
            env=args.env,
            output_file=args.output,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()
