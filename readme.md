
# 🚀 WebArena World Model Agent

带有memory joint optimization的版本，训练相当稳定，但训练结果暂时不佳，正在优化。
参考 [rLLM](https://github.com/rllm-org/rllm) 配置环境后，即可运行本项目。

## 🔧 使用步骤

1. 部署 vLLM serve 作为 World Model 服务  
2. 修改脚本中的相关路径与参数（如 `WANDB_KEY`、模型路径等）  
3. 运行训练脚本：


```bash
bash scripts/agent/webarena/gspo.sh
```

📌 生成数据：

```bash
bash world/run.sh  （参考rllm这个branch https://github.com/jadeleiyu/MBRL-Agent/tree/rllm）
```


## 关于多机大规模训练

（参考rllm这个branch https://github.com/jadeleiyu/MBRL-Agent/tree/rllm）

