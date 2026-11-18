#!/usr/bin/env bash
set -euo pipefail #命令出错时及时停止

cd /MiniOneRec

export CUDA_VISIBLE_DEVICES=0 #使用第一张显卡
export HF_HOME=/root/autodl-tmp/cache/huggingface #Hugging Face 缓存放数据盘
export HF_HUB_OFFLINE=1 #使用本地模型，不联网下载
# 在单机上启动一个训练进程
# 读取已经下载好的 Qwen
# 指定训练集、验证集
# 指定 SID 映射、商品信息
# 将训练结果保存到数据盘


torchrun --standalone --nproc_per_node=1 sft.py \
  --base_model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --train_file /MiniOneRec/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --eval_file /MiniOneRec/data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --sid_index_path /MiniOneRec/data/Amazon/index/Industrial_and_Scientific.index.json \
  --item_meta_path /MiniOneRec/data/Amazon/index/Industrial_and_Scientific.item.json \
  --output_dir /root/autodl-tmp/outputs/sft_smoke \
  --category Industrial_and_Scientific \
  --batch_size 64 \
  --micro_batch_size 2 \
  --sample 512 \
  --num_epochs 1 \
  --cutoff_len 512 \
  --train_from_scratch False \
  --freeze_LLM False \
  --seed 42




# export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
# # Office_Products, Industrial_and_Scientific
# for category in "Industrial_and_Scientific"; do
#     train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
#     eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
#     test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
#     info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
#     echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
#     torchrun --nproc_per_node 8 \
#             sft.py \
#             --base_model your_model_path \
#             --batch_size 1024 \
#             --micro_batch_size 16 \
#             --train_file ${train_file} \
#             --eval_file ${eval_file} \
#             --output_dir output_dir/xxx \
#             --wandb_project wandb_proj \
#             --wandb_run_name wandb_name \
#             --category ${category} \
#             --train_from_scratch False \
#             --seed 42 \
#             --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
#             --item_meta_path ./data/Amazon/index//Industrial_and_Scientific.item.json \
#             --freeze_LLM False
# done