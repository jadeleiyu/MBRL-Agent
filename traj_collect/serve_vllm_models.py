import submitit
import os
import datetime
import argparse
import yaml

# import sys
# sys.path.append(os.getcwd()) 
# from utils import paths_to_models

class Trainer:
    def __init__(self, output_dir, config, SCRIPT):
        self.cwd = config.get("cwd", "")
        self.conda_env_name = config.get("conda_env_name", "")
        self.conda_path = config.get("conda_path", "")
        self.output_dir = output_dir
        self.args = config.get("args", {})
        self.script = SCRIPT

    def create_cmd(self):
        cmd = f"""
source {self.conda_path}/etc/profile.d/conda.sh
conda activate {self.conda_env_name}
hash -r

echo "Using python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda envs: $(conda env list)"

{self.script}
"""
        print(cmd)
        return cmd

    def __call__(self):
        import os
        import subprocess
        os.chdir(self.cwd)
        cmd = self.create_cmd()
        subprocess.run(cmd, shell=True, check=True, executable="/bin/zsh")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Submitting Inference Job")
    parser.add_argument(
        "--config", type=str, default="configs/vllm.yaml", help="Path to the YAML config file"
    )
    parser.add_argument(
        "--partition", type=str, default="learnfair", help="Slurm partition to use"
    )
    parser.add_argument(
        "--timeout", type=int, default=4320, help="Timeout in minutes (default: 72 hours)"
    )
    parser.add_argument(
        "--model", type=str, default="", help="model"
    )
    parser.add_argument(
        "--port", type=str, default="8000", help="model"
    )
    parser.add_argument(
        "--priority", type=str, default="multimodal-reasoning_high", help="model key"
        # "--priority", type=str, default="lowest", help="model key"
    )
    parser.add_argument(
        "--model_download-dir", type=str, default="/checkpoint/multimodal-reasoning/jadeleiyu/huggingface", help="directory for huggingface model to be downloaded"
    )
    parser.add_argument(
        "--n_gpu_per_model", type=int, default=4, help="number of GPUs per served vllm model"
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_run_output_dir(model_key):  
    time_stamp = datetime.datetime.now().strftime("%m%d-%H%M")
    log_output_dir = f'logs/vllm/{model_key}-{time_stamp}'
    return log_output_dir


if __name__ == "__main__":

    args = parse_args()
    # Load configuration
    config = load_config(args.config)

    # Set model name
    MODEL = args.model
    model_key = MODEL.split('/')[-1]
    model_download_dir = args.model_download_dir

    # Make directories
    log_output_dir = get_run_output_dir(model_key)
    os.makedirs(log_output_dir, exist_ok=True)

    # Set parameters for the job
    SCRIPT = f"""vllm serve {MODEL} --port {args.port} --host 0.0.0.0 \
--tensor-parallel-size {args.n_gpu_per_model} --max_model_len 32768 \
--download-dir={model_download_dir}"""

    # Set up the executor
    executor = submitit.AutoExecutor(folder=log_output_dir)
    executor.update_parameters(
        name=model_key+":"+str(args.port),  # Job name
        mem_gb=1024,
        gpus_per_node=args.n_gpu_per_model,
        cpus_per_task=64,
        nodes=1,
        timeout_min=args.timeout,
        slurm_qos=args.priority,
        slurm_account="multimodal-reasoning")

    # Submit the job
    job = executor.submit(Trainer(log_output_dir, config, SCRIPT))

    print(f"Submitted job with ID: {job.job_id}, NAME: {model_key}")
    print(f'Output directory: {log_output_dir}')


