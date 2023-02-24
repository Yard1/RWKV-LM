#/bin/bash

cp -f /home/ray/default/RWKV-v4neo/20B_tokenizer.json /mnt/cluster_storage/20B_tokenizer.json
python run_on_every_node.py mount_nvme
python run_on_every_node.py download_model "RWKV-4-Pile-1B5"
# Download is done through https://github.com/Yard1/gpt-neox/blob/main/prepare_data.py
python run_on_every_node.py download_pile "pile_subset"