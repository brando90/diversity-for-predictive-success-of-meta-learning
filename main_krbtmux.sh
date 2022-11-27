#!/bin/bash
#!/usr/bin/expect -f
# https://unix.stackexchange.com/questions/724902/how-does-one-send-new-commands-to-run-to-an-already-running-nohup-process-e-g-r

# - CAREFUL, if a job is already running it could do damage to it, rm reauth process, qian doesn't do it so skip it
# top -u brando9
# pkill -9 tmux -u brando9; pkill -9 krbtmux -u brando9; pkill -9 reauth -u brando9; pkill -9 python -u brando9; pkill -9 wandb-service* -u brando9;
# krbtmux
# reauth
# sh main_krbtmux.sh

# - set up this main sh script
export RUN_PWD=$(pwd)

# stty -echo or stty echo
source /etc/bashrc
source /etc/profile
source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile
source ~/.bashrc.user
echo HOME = $HOME
# since snap .bash.user cd's me into HOME at dfs
cd $RUN_PWD
echo RUN_PWD = $RUN_PWD
realpath .

source cuda11.1

conda init bash
conda activate metalearning_gpu

# - get a job id
export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))")
echo SLURM_JOBID = $SLURM_JOBID

export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
export ERR_FILE=$PWD/main.sh.e$SLURM_JOBID
#export WANDB_DIR=$HOME
mkdir $LOCAL_MACHINE_PWD
export WANDB_DIR=$LOCAL_MACHINE_PWD

echo WANDB_DIR = $WANDB_DIR
echo OUT_FILE = $OUT_FILE
echo ERR_FILE = $ERR_FILE

export CUDA_VISIBLE_DEVICES=3
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
python -c "import torch; print(torch.cuda.get_device_name(0));"

# - 5CNN 4 filters
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_hdb1_5cnn_adam_cl_filter_size --filter_size 4 > $OUT_FILE 2> $ERR_FILE &

# - vit
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name vit_mi_fo_maml_rfs_adam_cl_100k > $OUT_FILE 2> $ERR_FILE &

# - delauny div
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_delauny > $OUT_FILE 2> $ERR_FILE &

# - hdb1 div
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_hdb1_mio > $OUT_FILE 2> $ERR_FILE &


# -- other option is to run `echo $SU_PASSWORD | /afs/cs/software/bin/reauth` inside of python, right?
export JOB_PID=$!
echo $OUT_FILE
echo $ERR_FILE
echo JOB_PID = $JOB_PID
echo SLURM_JOBID = $SLURM_JOBID

# - Done
echo "Done with the dispatching (daemon) sh script"
