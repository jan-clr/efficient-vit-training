#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --partition=a100
#SBATCH --time=24:00:00

hostname
whoami
echo $CUDA_VISIBLE_DEVICES

module load cuda/11.8.0
. .venv/bin/activate

python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 53533 --use_env main.py --model deit_small_patch16_224 --batch-size 64 --data-path $WORK/datasets/imagenet/ --output_dir $WORK/vit_runs --lr 1e-3 --localvit --localvit-act 'hs' --is-multisize --init-size 32 --epoch-step 5 --patch-step 2 --eval-on-final-size
echo "DONE"
