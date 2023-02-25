#/bin/bash

pip uninstall -y deepspeed && pip install -U --user deepspeed

cd ./RWKV-v4neo
# WANDB IS NOT YET WORKING!
python train.py --load_model "/nvme/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040.pth" --wandb "" --proj_dir "/nvme/out" \
--data_file "/nvme/data/pile/pile_00/pile_00_text_document" --data_type "binidx" --vocab_size 50277 \
--ctx_len 1024 --epoch_steps 200 --epoch_count 2 --epoch_begin 0 --epoch_save 1 \
--micro_bsz 20 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 \
--lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
--accelerator gpu --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 1