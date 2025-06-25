#!/bin/bash -l

cd $HOME/repos/efficient-vit-training/

mkdir -p logs

timestamp=$(date +%Y%m%d%H%M%S)

runname="vit_training_${timestamp}"

sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL varsize_1gpu_best_tinyimnet.sh
