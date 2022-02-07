#!/bin/bash
# pkill -9 python
# pkill -U miranda9

# -- setup up for condor_submit background script in vision-cluster
export HOME=/home/miranda9
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
export SLURM_JOBID=$(((RANDOM)))
echo SLURM_JOBID=$SLURM_JOBID
export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
echo OUT_FILE=$OUT_FILE

#pip install wandb --upgrade

# - run experiment
#python -u ~/automl-meta-learning/automl-proj-div_src/experiments/meta_learning/_main_metalearning.py
#python -u ~/automl-meta-learning/results_plots_sl_vs_ml/fall2021/main_distance_sl_vs_maml.py

# python -u ~/diversity-for-predictive-success-of-meta-learning/src/diversity_src/experiment_mains/_main_metalearning.py


# - new code
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name resnet12_rfs_cifarfs
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name manual_load_cifarfs_resnet12rfs_train_until_convergence

#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_metalearning2.py --manual_loads_name manual_load_mi_resnet12rfs_maml

#python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k
#python -m torch.distributed.run --nproc_per_node=8 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_4CNNl2l_cifarfs_rfs_adam_cl_70k

python -m torch.distributed.run --nproc_per_node=32 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_4CNNl2l_cifarfs_rfs_adam_cl_70k
echo pid = $!

echo "Done with bash script (experiment or dispatched daemon experiments). "
