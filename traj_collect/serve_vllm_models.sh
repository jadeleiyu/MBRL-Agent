MODELS=(

# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"


# "openai/gpt-oss-120b"
# "openai/gpt-oss-120b"
# "openai/gpt-oss-120b"
"openai/gpt-oss-120b"

)
N_GPU_PER_MODEL=4

export HF_HUB_CACHE=/checkpoint/multimodal-reasoning/jadeleiyu/huggingface

for MODEL in "${MODELS[@]}"
do
    echo $MODEL
    python serve_vllm_models.py\
        --model=$MODEL \
        --port=8000 \
        --n_gpu_per_model=$N_GPU_PER_MODEL \
        --priority=lowest
done

# bash serve_vllm_models.sh
