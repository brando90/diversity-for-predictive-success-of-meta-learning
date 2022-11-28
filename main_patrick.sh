#!/bin/bash

# -- setup up for condor_submit background script in vision-cluster
export HOME=/home/pzy2
# to have modules work and the conda command work
source /etc/bashrc
source /etc/profile
source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile


module load gcc/9.2.0
module load cuda-toolkit/11.1
nvcc --version

# some quick checks
conda activate metalearning3.9 #metalearning_gpu
python -c "import uutils; print(uutils); uutils.hello()"
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# - set wandb path
export WANDB_DIR=$HOME/wandb_dir
echo WANDB_DIR = $WANDB_DIR

# gpu name
python -c "import torch; print(torch.cuda.get_device_name(0));"

#echo ---- Running your python main ----
echo PWD = $PWD

# -- Run Experiment (TODO? delete files from experiment_mains that arent related)
# - Div of MDS
#python -u ~/example.py --data_path /shared/rsaas/pzy2/records
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/pytorch_meta_dataset/metadataset_task2vec_div.py
# - USL
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/pytorch_meta_dataset/metadataset_usl.py
# - MAML
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/pytorch_meta_dataset/maml_metadataset.py

echo "---> Done with bash script (experiment or dispatched daemon experiments). "
