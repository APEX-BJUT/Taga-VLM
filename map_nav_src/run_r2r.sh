DATA_ROOT=/root/liujiaxing/tagavlm_infer/TagaVLM_infer_data

train_alg=dagger

features=vitbase
ft_dim=768
obj_features=vitbase
obj_ft_dim=768
# /root/miniconda3/envs/llava-duet/lib/python3.9/site-packages
ngpus=1
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-init.aug.45k

outdir=/root/liujiaxing/tagavlm_infer/map_nav_src/output/sample_trainer_test_epoch=3-sride=3-Scheduler-FSDP-2-20
flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 1
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

# train
# CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
#       --tokenizer bert \
#       --bert_ckpt_file ../datasets/R2R/trained_models/best_val_unseen \
#       --eval_first

# test
CUDA_VISIBLE_DEVICES='2' python main_nav_llava.py $flag  \
      --tokenizer bert \
      --test --submit