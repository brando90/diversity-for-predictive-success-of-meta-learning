#!/bin/bash
# pkill -9 python

# -- setup up for condor_submit background script in vision-cluster
#source /etc/bashrc
#source /etc/profile
#source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile
source ~/.bashrc.user
echo HOME = $HOME

/afs/cs/software/bin/reauth
# type password here

#module load gcc/9.2.0
source cuda11.1

# activate conda
conda init bash
conda activate metalearning_gpu

# some quick checks
nvcc --version
hostname
which python
python -c "import uutils; print(uutils); uutils.hello()"
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# - test shared
pwd .
realpath .

# - check wanbd dir
echo WANDB_DIR = $WANDB_DIR
echo TEMP = $TEMP

# -- start experiment run
echo -- Start my submission file

# - choose GPU
#export CUDA_VISIBLE_DEVICES=$(((RANDOM%8)))
#export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=2
#export CUDA_VISIBLE_DEVICES=3
#export CUDA_VISIBLE_DEVICES=4
#export CUDA_VISIBLE_DEVICES=5
#export CUDA_VISIBLE_DEVICES=6
#export CUDA_VISIBLE_DEVICES=7
#export CUDA_VISIBLE_DEVICES=4,5,6,7
#export CUDA_VISIBLE_DEVICES=4,5,7
#export CUDA_VISIBLE_DEVICES=4,5,7
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=1,2,3
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,6,7

# - set experiment id
#python -c "import random;print(random.randint(0, 1_000_000))"
#export SLURM_JOBID=$(((RANDOM)))
export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))")
echo SLURM_JOBID = $SLURM_JOBID

#export OUT_FILE="$($PWD)/main.sh.o$($SLURM_JOBID)_gpu$($CUDA_VISIBLE_DEVICES)_$(hostname)"
export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
echo OUT_FILE = $OUT_FILE

#export OUT_FILE="$($PWD)/main.sh.err$($SLURM_JOBID)_gpu$($CUDA_VISIBLE_DEVICES)_$(hostname)"
export ERR_FILE=$PWD/main.sh.err$SLURM_JOBID
echo ERR_FILE = $ERR_FILE

echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
# gpu name & number of gpus
python -c "import torch; print(torch.cuda.get_device_name(0));"
python -c "import uutils; uutils.torch_uu.gpu_name_otherwise_cpu(print_to_stdout=True);"

# -- Run Experiment
echo ---- Running your python main.py file ----

# hdb1 scaling expts with 5CNN
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_hdb1_adam_cs_filter_size
#nohup python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_hdb1_adam_cs_filter_size > $OUT_FILE 2> $ERR_FILE
nohup python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_hdb1_adam_cs_filter_size > $OUT_FILE 2> $ERR_FILE &

# -- echo useful info, like process id/pid
echo pid = $!
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
echo SLURM_JOBID = $SLURM_JOBID

# - Done
echo "Done with bash script (experiment or dispatched daemon experiments). "

#pip install wandb --upgrade
#df -h
