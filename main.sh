#!/bin/bash
# pkill -9 python
# pkill -U miranda9
# chmod a+x ~/diversity-for-predictive-success-of-meta-learning/main.sh

# -- setup up for condor_submit background script in vision-cluster
 export HOME=/home/miranda9
#export HOME=/shared/rsaas/miranda9
# to have modules work and the conda command work
source /etc/bashrc
source /etc/profile
source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile

module load gcc/9.2.0
#module load cuda-toolkit/10.2
module load cuda-toolkit/11.1

# some quick checks
#conda activate synthesis
conda activate metalearning_gpu
#export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=1
nvcc --version
hostname
which python
python -c "import uutils; print(uutils); uutils.hello()"
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo ---- Running your python main ----

# set experiment id
#export SLURM_JOBID=$(((RANDOM)))
#echo SLURM_JOBID=$SLURM_JOBID
#export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
#echo OUT_FILE=$OUT_FILE

# - test shared
pwd .
realpath .

#export WANDB_DIR=~/tmp
#export WANDB_DIR=/shared/rsaas/tmp
#echo WANDB_DIR

# -- Run Experiment
# - SL
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_cl_32_filter_size

# - MAML
# python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_mi_rfs_adam_cl_100k
#python -m torch.distributed.run --nproc_per_node=2 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_mi_rfs_adam_cl_100k

# no CA sched hdb1
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_hdb1_100k_adam_no_scheduler
#python -m torch.distributed.run --nproc_per_node=2 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_hdb1_100k_adam_no_scheduler
#python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_hdb1_100k_adam_no_scheduler

# yes CA sched hdb1
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler
#python -m torch.distributed.run --nproc_per_node=2 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler
#python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler

# yes CA first order hdb1
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler_first_order
#python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler_first_order

# vit CA fo mi
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name vit_mi_fo_maml_rfs_adam_cl_100k
#python -u /shared/rsaas/miranda9/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name vit_mi_fo_maml_rfs_adam_cl_100k
python ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name vit_mi_fo_maml_rfs_adam_cl_100k

# - Data analysis
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main2_distance_sl_vs_maml.py
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/_main_distance_sl_vs_maml.py
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py

#pip install wandb --upgrade

echo "Done with bash script (experiment or dispatched daemon experiments). "
