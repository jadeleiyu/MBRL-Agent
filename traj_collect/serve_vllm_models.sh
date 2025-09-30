MODELS=(

"/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
"/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
"/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"
# "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-120b-llamafactory-merged-500"

)
N_GPU_PER_MODEL=4

for MODEL in "${MODELS[@]}"
do
    echo $MODEL
    python serve_vllm_models.py\
        --model=$MODEL \
        --port=8000 \
        --n_gpu_per_model=$N_GPU_PER_MODEL
done

# bash serve_vllm_models.sh
