#!/bin/bash
#SBATCH --job-name=vllm_generator
#SBATCH --nodes=1
#SBATCH --gres=gpu:8               # adjust
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --qos=multimodal-reasoning_high
#SBATCH --time=24:00:00
#SBATCH -o ./vllm_log/vllm_generator.%j.out
#SBATCH -e ./vllm_log/vllm_generator.%j.err

# --- config you set before sbatch ---
MODEL_ID=${MODEL_ID:-openai/gpt-oss-20b}   # your M
PORT=${PORT:-8000}
VLLM_API_KEY=${VLLM_API_KEY:-mbrl_agent}               # any non-empty string
DISCOVERY_JSON=${DISCOVERY_JSON:-/home/jadeleiyu/projects/mbrl_agent/train/vllm_log/vllm_discovery.json}  # shared FS path
# HF_HOME=${HF_HOME:-/scratch/hf}                          # optional cache

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate mbrl_agent

module load cuda || true
# export HF_HOME
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# Helpful Slurm envs for node identity
NODE_HOST="${SLURMD_NODENAME:-$(hostname -s)}"  # Slurm sets SLURMD_NODENAME on compute nodes. :contentReference[oaicite:4]{index=4}

# 1) start Ray HEAD with a custom resource tag so we can place actors on this node
ray start --head \
  --port=6379 \
  --ray-client-server-port=10001 \
  --resources='{"vllm_node": 1}' \
  --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265

# 2) launch vLLM OpenAI-compatible server (tensor-parallel across this node's GPUs)
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --dtype auto \
  --tensor-parallel-size "${SLURM_GPUS_ON_NODE:-8}" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --api-key "$VLLM_API_KEY" \
  --enable-lora \
  > ./vllm_log/vllm_server.log 2>&1 &

# Wait for port to be ready (simple loop)
for i in {1..60}; do
  curl -s "http://localhost:${PORT}/v1/models" -H "Authorization: Bearer ${VLLM_API_KEY}" >/dev/null && break
  sleep 2
done

# 3) write discovery info for Node B
ENDPOINT="http://${NODE_HOST}:${PORT}/v1"
RAY_ADDR="ray://${NODE_HOST}:10001"
echo "{\"endpoint\":\"${ENDPOINT}\",\"ray_address\":\"${RAY_ADDR}\",\"model_id\":\"${MODEL_ID}\"}" > "${DISCOVERY_JSON}"
echo "Discovery written to: ${DISCOVERY_JSON}"
echo "vLLM endpoint: ${ENDPOINT}"
echo "Ray Client:    ${RAY_ADDR}"

# keep job alive (vLLM is in the background)
wait
