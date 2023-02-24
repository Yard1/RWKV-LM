# RWKV on Ray Train

RWKV-v4neo on Ray Train.

Cluster environment:
![Cluster environment](cluster_env.png)

Original readme in `ORIGINAL_README.md`.

## Instructions

First, run `prepare.sh`, which will mount NVMe drives on GPU nodes, download the pretrainted RWKV-4-Pile-1B5 model and then download a subset of the Pile dataset (1/16) and tokenize it.

Afterwards, run `train.sh`.