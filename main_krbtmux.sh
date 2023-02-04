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
#ssh brando9@ampere3.stanford.edu
#ssh brando9@ampere4.stanford.edu

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
echo $OUT_FILE; echo $ERR_FILE

# - mds
# C-b d (C=Ctl not command, similar tmux detach)

ssh brando9@ampere1.stanford.edu
ssh brando9@ampere2.stanford.edu
ssh brando9@ampere3.stanford.edu
ssh brando9@ampere4.stanford.edu

tput rmcup

krbtmux
reauth

source $AFS/.bashrc.lfs
conda activate mds_env_gpu
#conda activate metalearning_gpu
export CUDA_VISIBLE_DEVICES=1; export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))")
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES; echo SLURM_JOBID = $SLURM_JOBID; echo hostname = $(hostname)
ulimit -n 120000; ulimit -Sn; ulimit -Hn
nvidia-smi; ps -up `nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//'`; hostname

tmux new -s gpu0
tmux new -s gpu1
tmux new -s gpu4
tmux new -s gpu6
tmux new -s gpu7

tmux new -s mds0_maml_resnet50rfs
tmux new -s mds1_maml_resnet50rfs
tmux new -s mds0_maml_resnet50rfs_smart

tmux new -s mds0_usl_resnet50rfs
tmux new -s mds1_usl_resnet50rfs
tmux new -s mds0_usl_resnet50rfs_smart

#tmux new -s div_hdb4_micod
#tmux new -s div_hdb4_micod2

#tmux new -s hdb4_usl_its

#tmux new -s hdb4_maml_its_sched
#tmux new -s hdb4_maml_its_sched2

#tmux new -s hdb1_stats_analysis
#tmux new -s hdb4_stats_analysis

#tmux new -s rand

#bash ~/diversity-for-predictive-success-of-meta-learning/main_krbtmux.sh

# - div
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_mds \
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_hdb4_micod --model_option resnet18_pretrained_imagenet \
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_hdb4_micod --model_option resnet34_pretrained_imagenet \

# - mds maml
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_maml_torchmeta.py --manual_loads_name mds_resnet_maml_adam_scheduler --model_option resnet18_rfs --data_path $HOME/data/mds/records/
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_maml_torchmeta.py --manual_loads_name mds_resnet_maml_adam_scheduler --model_option resnet50_rfs --data_path $HOME/data/mds/records/
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_maml_torchmeta.py --manual_loads_name mds_maml --model_option resnet50_rfs --data_path $HOME/data/mds/records/

# - mds usl
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name mds_resnet_usl_adam_scheduler --model_option resnet18_rfs --data_path $HOME/data/mds/records/
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name mds_resnet_usl_adam_scheduler --model_option resnet50_rfs --data_path $HOME/data/mds/records/
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name mds_usl --model_option resnet50_rfs --data_path $HOME/data/mds/records/

# - hdb4 micod usl
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_resnet_rfs_adam_cl_its --model_option resnet12_rfs
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_resnet_rfs_adam_cl_train_to_convergence --model_option resnet12_rfs
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_resnet_rfs_log_more_often_0p9_acc_reached --model_option resnet12_rfs

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_convg_reached_log_ckpt_more --filter_size 4 --model_option 5CNN_opt_as_model_for_few_shot

# - hdb4 micod maml
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod_resnet_rfs_scheduler_its --model_option resnet12_rfs
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod_resnet_rfs_scheduler_train_to_convergence --model_option resnet12_rfs

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod_log_more_often_convg --filter_size 4 --model_option 5CNN_opt_as_model_for_few_shot

# - performance comp usl vs maml
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_experiment_analysis_sl_vs_maml_performance_comp_distance.py --manual_loads_name resnet12rfs_hdb1_mio
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_experiment_analysis_sl_vs_maml_performance_comp_distance.py --manual_loads_name hdb4_micod

# -- other option is to run `echo $SU_PASSWORD | /afs/cs/software/bin/reauth` inside of python, right?
export JOB_PID=$!
echo $OUT_FILE
echo $ERR_FILE
echo JOB_PID = $JOB_PID
echo SLURM_JOBID = $SLURM_JOBID

# - Done
echo "Done with the dispatching (daemon) sh script"
