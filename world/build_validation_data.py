#!/usr/bin/env python3
"""
构建validation数据，基于nnetnav数据集
按照现有数据集构建模式，将数据分为train/val两部分
"""

import json
import os
import argparse
import random
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


def create_train_val_split(env: str = "live", 
                          output_dir: str = "data",
                          train_ratio: float = 0.8,
                          max_samples: int = 1000,
                          random_seed: int = 42) -> None:
    """
    创建train/val数据分割，按照现有数据集构建模式
    
    Args:
        env: 环境名称 (live, wa等)
        output_dir: 输出目录
        train_ratio: 训练数据比例，默认为0.8
        max_samples: 最大样本数
        random_seed: 随机种子
    """
    print(f"加载数据集: stanfordnlp/nnetnav-{env}")
    ds = load_dataset(f"stanfordnlp/nnetnav-{env}", split='train')
    
    # 限制样本数
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 创建索引列表并打乱
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    # 计算分割点
    train_size = int(train_ratio * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    print(f"总样本数: {len(ds)}")
    print(f"训练样本数: {len(train_indices)}")
    print(f"验证样本数: {len(val_indices)}")
    
    # 过滤函数：跳过 axtree_txt 超过 30000 字符的样本
    def _within_ax_limit(acc_tree: str) -> bool:
        try:
            return not (isinstance(acc_tree, str) and len(acc_tree) > 30000)
        except Exception:
            return True

    # 处理训练数据
    train_data = []
    for i, idx in enumerate(train_indices):
        try:
            sample = ds[idx]
            task_info = parse_nnetnav_task(sample)
            instruction = task_info["instruction"]
            acc_tree = task_info["acc_tree"]
            url = task_info["url"]
            if not _within_ax_limit(acc_tree):
                continue
            
            # 构建WebArena风格的prompt
            messages = build_webarena_prompt(instruction, acc_tree, url)
            
            # 构建训练数据条目
            data_entry = {
                "data_source": f"stanfordnlp/nnetnav-{env}",
                "prompt": messages,
                "ability": "web",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": "train",
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
            
            train_data.append(data_entry)
            
        except Exception as e:
            print(f"处理训练样本 {i} 时出错: {e}")
            continue
    
    # 处理验证数据
    val_data = []
    for i, idx in enumerate(val_indices):
        try:
            sample = ds[idx]
            task_info = parse_nnetnav_task(sample)
            instruction = task_info["instruction"]
            acc_tree = task_info["acc_tree"]
            url = task_info["url"]
            if not _within_ax_limit(acc_tree):
                continue
            
            # 构建WebArena风格的prompt
            messages = build_webarena_prompt(instruction, acc_tree, url)
            
            # 构建验证数据条目
            data_entry = {
                "data_source": f"stanfordnlp/nnetnav-{env}",
                "prompt": messages,
                "ability": "web",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": "val",
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
            
            val_data.append(data_entry)
            
        except Exception as e:
            print(f"处理验证样本 {i} 时出错: {e}")
            continue
    
    # 保存数据
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练数据
    train_df = pd.DataFrame(train_data)
    train_file = os.path.join(output_dir, "train.parquet")
    train_df.to_parquet(train_file)
    print(f"训练数据已保存到: {train_file}")
    print(f"训练数据大小: {len(train_data)} 个样本")
    
    # 保存验证数据
    val_df = pd.DataFrame(val_data)
    val_file = os.path.join(output_dir, "test.parquet")  # 使用test.parquet作为val文件，与训练脚本一致
    val_df.to_parquet(val_file)
    print(f"验证数据已保存到: {val_file}")
    print(f"验证数据大小: {len(val_data)} 个样本")
    
    # 显示示例
    if train_data:
        print("\n训练数据示例:")
        example = train_data[0]
        print(f"Instruction: {example['extra_info']['instruction']}")
        print(f"URL: {example['extra_info']['url']}")
        print(f"Split: {example['extra_info']['split']}")
    
    if val_data:
        print("\n验证数据示例:")
        example = val_data[0]
        print(f"Instruction: {example['extra_info']['instruction']}")
        print(f"URL: {example['extra_info']['url']}")
        print(f"Split: {example['extra_info']['split']}")


def main():
    parser = argparse.ArgumentParser(description="构建train/val数据分割")
    parser.add_argument("--env", type=str, default="live", 
                       help="环境名称 (live, wa等)")
    parser.add_argument("--output_dir", type=str, default="data",
                       help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="训练数据比例 (默认: 0.8)")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="最大样本数")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    create_train_val_split(
        env=args.env,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        max_samples=args.max_samples,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
