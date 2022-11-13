#!/bin/bash
# pkill -9 python
# pkill -U miranda9
# chmod a+x ~/diversity-for-predictive-success-of-meta-learning/main.sh
# chmod a+x /shared/rsaas/miranda9/diversity-for-predictive-success-of-meta-learning/main.sh

# -- setup up for condor_submit background script in vision-cluster
export HOME=/home/miranda9
#export HOME=/shared/rsaas/miranda9
# to have modules work and the conda command work
source /etc/bashrc
source /etc/profile
source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile

pwd .
realpath .
hostname

module load gcc/9.2.0
module load cuda-toolkit/11.1
nvcc --version

# some quick checks
conda activate metalearning_gpu
which python
python -c "import uutils; print(uutils); uutils.hello()"
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# - set wandb path
#echo 'TEMP='
#echo $TEMP
#export WANDB_DIR=/shared/rsaas/miranda9/tmp
#export WANDB_DIR=$TEMP
export WANDB_DIR=$HOME/wandb_dir
echo WANDB_DIR = $WANDB_DIR

# gpu name
python -c "import torch; print(torch.cuda.get_device_name(0));"
python -c "import uutils; gpu_name_otherwise_cpu(print_to_stdout=True);"

echo ---- Running your python main ----
# -- Run Experiment
# - SL
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_cl_32_filter_size

#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_hdb1_rfs_resnet12rfs_adam_cl

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

# hdb1 from ckpt
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_hdb1_100k_adam_cosine_scheduler_first_order_from_ckpt

# vit CA fo mi
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name vit_mi_fo_maml_rfs_adam_cl_100k
#python -u /shared/rsaas/miranda9/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name vit_mi_fo_maml_rfs_adam_cl_100k
#python ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name vit_mi_fo_maml_rfs_adam_cl_100k

# hdb1 scaling expts with 5CNN
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_hdb1_adam_cs_filter_size

# mi scaling with 5cnn 128, 512
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_mi_adam_filter_size_128_filter_size
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_mi_adam_filter_size_512_filter_size

#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_cl_128_filter_size
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_cl_512_filter_size

# - Data analysis
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main2_distance_sl_vs_maml.py
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/_main_distance_sl_vs_maml.py
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py

#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_experiment_analysis_sl_vs_maml_performance_comp_distance.py

#rm -rf $HOME/wandb_dir

#pip install wandb --upgrade
echo "Done with bash script (experiment or dispatched daemon experiments). "
