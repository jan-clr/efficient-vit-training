#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --partition=v100
#SBATCH --time=24:00:00

hostname
whoami
echo $CUDA_VISIBLE_DEVICES

module load cuda/12.6.2
. .venv/bin/activate

python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 53533 --use_env main.py --model deit_small_patch16_224 --batch-size 64 --data-path $WORK/datasets/cifar/ --data-set CIFAR --output_dir $WORK/vit_cifar_12 --lr 1e-3 --localvit --localvit-act 'hs' --init-size 32 --epoch-step 5 --patch-step 2 --eval-on-final-size --epochs 1000 --depth 12
echo "DONE"
