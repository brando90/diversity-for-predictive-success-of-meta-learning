#!/bin/bash
#!/usr/bin/expect -f
# https://unix.stackexchange.com/questions/724902/how-does-one-send-new-commands-to-run-to-an-already-running-nohup-process-e-g-r

# - CAREFUL, if a job is already running it could do damage to it, rm reauth process, qian doesn't do it so skip it
# top -u brando9
#
# pkill -9 reauth -u brando9;
#
# pkill -9 tmux -u brando9; pkill -9 krbtmux -u brando9; pkill -9 reauth -u brando9; pkill -9 python -u brando9; pkill -9 wandb-service* -u brando9;
#
# pkill -9 python -u brando9; pkill -9 wandb-service* -u brando9;
#
# krbtmux
# reauth
# nvidia-smi
# sh main_krbtmux.sh
#
# tmux attach -t 0

# ssh brando9@hyperturing1.stanford.edu
# ssh brando9@hyperturing2.stanford.edu
# ssh brando9@turing1.stanford.edu
# ssh brando9@ampere1.stanford.edu

# - set up this main sh script
export RUN_PWD=$(pwd)

# stty -echo or stty echo
sh /etc/bashrc
sh /etc/profile
sh /etc/profile.d/modules.sh
sh ~/.bashrc
sh ~/.bash_profile
sh ~/.bashrc.user
echo HOME = $HOME
# since snap .bash.user cd's me into HOME at dfs
cd $RUN_PWD
echo RUN_PWD = $RUN_PWD
realpath .

# - https://ilwiki.stanford.edu/doku.php?id=hints:gpu
#sh cuda11.1
#source cuda11.1
#source cuda11.6
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
#export PATH=/usr/local/cuda-11.7/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
nvcc -V

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

#python -c "import torch; print(torch.cuda.get_device_name(0));"

# sh main_krbtmux.sh
# - 5CNN 4 filters
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_hdb1_5cnn_adam_cl_filter_size --filter_size 4 > $OUT_FILE 2> $ERR_FILE &

# - vit
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name vit_mi_fo_maml_rfs_adam_cl_100k > $OUT_FILE 2> $ERR_FILE &

# - delauny div
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_delauny > $OUT_FILE 2> $ERR_FILE &

# - hdb1 div
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_hdb1_mio > $OUT_FILE 2> $ERR_FILE &

# - hdb2 div
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_hdb2_cifo > $OUT_FILE 2> $ERR_FILE &

# - mi vs omni
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/diversity/task2vec_based_metrics/diversity_task2vec/mi_vs_omniglot_div.py > $OUT_FILE 2> $ERR_FILE &

# - mds div
conda activate mds_env_gpu
echo $OUT_FILE; echo $ERR_FILE
export CUDA_VISIBLE_DEVICES=3; echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES

# - mds maml
source $AFS/.bashrc.lfs
conda activate mds_env_gpu
#tmux new -s mds_maml
#tmux new -s mds_usl_resnet18rfs
#tmux new -s mds_maml_resnet50rfs
#tmux attach -t mds_maml

#bash ~/diversity-for-predictive-success-of-meta-learning/main_krbtmux.sh

#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_maml_torchmeta.py --manual_loads_name mds_resnet_maml_adam_scheduler --model_option resnet18_rfs --data_path $HOME/data/mds/records/ \
#    |& tee $OUT_FILE 2> $ERR_FILE
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_maml_torchmeta.py --manual_loads_name mds_resnet_maml_adam_scheduler --model_option resnet50_rfs --data_path $HOME/data/mds/records/ \
#    |& tee $OUT_FILE 2> $ERR_FILE

# - mds usl
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name mds_resnet_usl_adam_scheduler --model_option resnet18_rfs --data_path $HOME/data/mds/records/ \
    |& tee $OUT_FILE 2> $ERR_FILE
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name mds_resnet_usl_adam_scheduler --model_option resnet50_rfs --data_path $HOME/data/mds/records/ \
#    |& tee $OUT_FILE 2> $ERR_FILE

# - performance comp usl vs maml on hdb1
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_experiment_analysis_sl_vs_maml_performance_comp_distance.py > $OUT_FILE 2> $ERR_FILE &

# -- other option is to run `echo $SU_PASSWORD | /afs/cs/software/bin/reauth` inside of python, right?
export JOB_PID=$!
echo $OUT_FILE
echo $ERR_FILE
echo JOB_PID = $JOB_PID
echo SLURM_JOBID = $SLURM_JOBID

# - Done
echo "Done with the dispatching (daemon) sh script"
