#!/bin/bash

echo -- Start my submission file

export SLURM_JOBID=$(((RANDOM)))
echo SLURM_JOBID = $SLURM_JOBID

#export CUDA_VISIBLE_DEVICES=$(((RANDOM%8)))
#export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
#export CUDA_VISIBLE_DEVICES=2
#export CUDA_VISIBLE_DEVICES=3
#export CUDA_VISIBLE_DEVICES=4
#export CUDA_VISIBLE_DEVICES=5
#export CUDA_VISIBLE_DEVICES=6
#export CUDA_VISIBLE_DEVICES=7
#export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7

echo CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
echo torch.cuda.device_count is:
python -c "import torch; print(torch.cuda.device_count())"
echo ---- Running your python main ----

#export SLURM_JOBID=-1
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main2_metalearning.py --manual_loads_name manual_load_cifarfs_resnet12rfs_maml > $OUT_FILE &

# - SL
#export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_cifarfs_rfs_resnet12rfs_adam_cl_200 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_cl_200 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_resnet_rfs_mi_adam_cl_200 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_cifarfs_rfs_resnet12rfs_adam_cl_600 > $OUT_FILE &
#echo pid = $!
#echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
#echo SLURM_JOBID = $SLURM_JOBID

# - MAML
export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
#python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k > $OUT_FILE &
python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_mi_rfs_adam_cl_100k > $OUT_FILE &
echo pid = $!
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
echo SLURM_JOBID = $SLURM_JOBID

# - Data analysis
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main2_distance_sl_vs_maml.py

echo -- Done submitting job in dgx A100-SXM4-40G
