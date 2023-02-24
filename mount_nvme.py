import ray
import ray.util.scheduling_strategies
import subprocess
from pathlib import Path

def force_on_node(node_id: str, remote_func_or_actor_class):
    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id, soft=False
    )
    options = {"scheduling_strategy": scheduling_strategy}
    return remote_func_or_actor_class.options(**options)


def run_on_every_node(remote_func_or_actor_class, **remote_kwargs):
    refs = []
    for node in ray.nodes():
        if node["Alive"] and node["Resources"].get("GPU", None):
            refs.append(
                force_on_node(node["NodeID"], remote_func_or_actor_class).remote(**remote_kwargs)
            )
    return ray.get(refs)


@ray.remote(num_gpus=1)
def mount_nvme():
    subprocess.run(
        'drive_name="${1:-/dev/nvme1n1}"; mount_path="${2:-/nvme}"; set -x; sudo file -s "$drive_name"; sudo apt install xfsprogs -y; sudo mkfs -t xfs "$drive_name"; sudo mkdir "$mount_path" && sudo mount "$drive_name" "$mount_path" && sudo chown -R ray "$mount_path"', shell=True, check=True
    )

@ray.remote(num_gpus=1)
def test():
    subprocess.run("mountpoint /nvme", shell=True)

@ray.remote(num_gpus=1)
def prec():
   #subprocess.run("cp -r /mnt/cluster_storage/TorchTrainer_2023-02-21_15-52-40/TorchTrainer_d3578_00000_0_2023-02-21_15-52-41/checkpoint_000000 /nvme", shell=True, check=True)
    subprocess.run("cd /nvme; wget https://data.deepai.org/enwik8.zip && unzip enwik8.zip; rm enwik8.zip; ls", shell=True, check=True)

@ray.remote(num_gpus=1)
def download_model():
    base_model_name = "RWKV-4-Pile-1B5" #@param ["RWKV-4-Pile-1B5", "RWKV-4-Pile-430M", "RWKV-4-Pile-169M"]
    base_model_url = f"https://huggingface.co/BlinkDL/{base_model_name.lower()}"
   #subprocess.run("cp -r /mnt/cluster_storage/TorchTrainer_2023-02-21_15-52-40/TorchTrainer_d3578_00000_0_2023-02-21_15-52-41/checkpoint_000000 /nvme", shell=True, check=True)
    subprocess.run(f"cd /nvme; git lfs clone {base_model_url}; ls '{base_model_name.lower()}'", shell=True, check=True)

@ray.remote(num_gpus=1)
def download_pile():
    subprocess.run(
       "rm -rf /nvme/data/pile/; rm -rf ~/gpt-neox", shell=True, check=True
    )
    subprocess.run(
       "cd ~/; git clone https://github.com/Yard1/gpt-neox.git; cd gpt-neox; python prepare_data.py europarl -d /nvme/data/pile -t HFTokenizer --vocab-file '/mnt/cluster_storage/20B_tokenizer.json'", shell=True, check=True
    )
if __name__ == "__main__":
    ray.init()
    run_on_every_node(download_pile)

