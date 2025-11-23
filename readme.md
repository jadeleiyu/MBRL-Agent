
# 🚀 WebArena World Model Agent

适配到了rLLM全新更新的v0.2版本，并新增相关算法和代码实现，训练相当稳定。
参考 [rLLM](https://github.com/rllm-org/rllm) 配置环境后，即可运行本项目。

## 🔧 使用步骤

1. 部署 vLLM serve 作为 World Model 服务  
2. 修改脚本中的相关路径与参数（如 `WANDB_KEY`、模型路径等）  
3. 运行训练脚本：

GRPO：
```bash
bash scripts/agent/webarena/run_webarena_world_model.sh
````

DAPO：

```bash
bash scripts/agent/webarena/run_webarena_world_model_dapo.sh
```

GSPO：

```bash
bash scripts/agent/webarena/gspo.sh
```

**本脚本当前已更新为带有标准答案的版本，通过设置STANDARD_ANSWER_K=${STANDARD_ANSWER_K:-1}，选择每隔k-1步，每条数据，用标准答案随机替换rollout group中的一条，reward为1。**

**目前为5步以内的数据，初步观察成功率较高，可能要生成稍微长点的数据。**

📌 生成数据：

```bash
bash world/run.sh
```

## ✅ 当前状态

目前版本训练相当稳定。

| 方法   | Epoch | 权重链接                                                                                                                                       |
| ---- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| GRPO | 2     | [https://huggingface.co/DearSloth/Llama3.1-8b-GRPO-rllm0.2-WA-2epoch](https://huggingface.co/DearSloth/Llama3.1-8b-GRPO-rllm0.2-WA-2epoch) |
| DAPO | 2     | [https://huggingface.co/DearSloth/Llama3.1-8b-DAPO-rllm0.2-WA-2epoch](https://huggingface.co/DearSloth/Llama3.1-8b-DAPO-rllm0.2-WA-2epoch) |
| GSPO | 3     | [https://huggingface.co/DearSloth/Llama3.1-8b-GSPO-rllm0.2-WA-3epoch](https://huggingface.co/DearSloth/Llama3.1-8b-GSPO-rllm0.2-WA-3epoch) |

体感：GSPO 效果更好，但收敛稍慢。

## 关于多机大规模训练

1. 先生成数据 （`bash world/run.sh`，需修改相关参数），如果有条件，可先都rollout一遍，选择模型有成功又失败的数据来train。
2. 推荐从gspo或grpo开始（`multi_node_gspo.sh`,`multi_node_grpo.sh`）
3. 需要先部署world model，然后在脚本里修改url等，在当前setting下，推荐world model和train model的node比例为 1:2 or 1:1，也就是1个node部署，1或2node训练。
4. 当前单机训练无任何问题，multi_node脚本记得修改save/test freq等相关参数，多机参数由于显卡有限尚未测试，但应该问题不大。


