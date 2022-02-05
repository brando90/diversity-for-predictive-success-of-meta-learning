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
export CUDA_VISIBLE_DEVICES=1,7,6,5,4,3,2,0

echo CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
echo torch.cuda.device_count is:
python -c "import torch; print(torch.cuda.device_count())"
echo ---- Running your python main ----

#export SLURM_JOBID=-1
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_metalearning2.py --manual_loads_name manual_load_cifarfs_resnet12rfs_maml > $OUT_FILE &

#export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name manual_load_cifarfs_resnet12rfs_train_until_convergence > $OUT_FILE &
#echo pid = $!
#echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
#echo SLURM_JOBID = $SLURM_JOBID

export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
##python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_metalearning2.py --manual_loads_name manual_load_cifarfs_resnet12rfs_maml_official_correct_fo > $OUT_FILE &
##python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_metalearning2.py --manual_loads_name manual_load_cifarfs_resnet12rfs_maml_official_correct_fo_adam_no_scheduler > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_metalearning2.py --manual_loads_name manual_load_cifarfs_resnet12rfs_maml_ho_adam_simple_cosine_annealing > $OUT_FILE &
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_metalearning2.py --manual_loads_name manual_load_mi_resnet12rfs_maml_ho_adam_simple_cosine_annealing > $OUT_FILE &
echo pid = $!
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
echo SLURM_JOBID = $SLURM_JOBID

echo -- Done submitting job in dgx A100-SXM4-40G
