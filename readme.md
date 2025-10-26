
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

## ⚠️ 目前的问题

1、卡不太够，若其他老师同学有卡可以先test看看效果 
2、训练数据量暂时只有不到1k，因为训练较为耗时，1epoch1.5天的样子，如果卡多可能可以尝试多机训练（目前我自己的8卡train， 8卡部署world model），并且2或3epoch感觉远远不够 
3、Policy Model同时作为Memory module进行joint policy optimization的想法，正在训练尝试


