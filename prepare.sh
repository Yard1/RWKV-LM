#/bin/bash

python run_on_every_node.py mount_nvme
python run_on_every_node.py download_model "RWKV-4-Pile-1B5"
python run_on_every_node.py download_pile "pile_subset"