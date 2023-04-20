#!/bin/bash
#!/usr/bin/expect -f
# https://unix.stackexchange.com/questions/724902/how-does-one-send-new-commands-to-run-to-an-already-running-nohup-process-e-g-r

# - CAREFUL, if a job is already running it could do damage to it, rm reauth process, qian doesn't do it so skip it
# top -u brando9
#
# pkill -9 reauth -u brando9;
#
#pkill -9 tmux -u brando9; pkill -9 krbtmux -u brando9; pkill -9 reauth -u brando9; pkill -9 python -u brando9; pkill -9 wandb-service* -u brando9;
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
# ssh brando9@ampere2.stanford.edu
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
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_hdb1_mio

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

ssh brando9@hyperturing1.stanford.edu
ssh brando9@hyperturing2.stanford.edu

ssh brando9@mercury1.stanford.edu
ssh brando9@mercury2.stanford.edu

tput rmcup

krbtmux
reauth

# answer: https://www.reddit.com/r/HPC/comments/10x9w6x/comment/j7sg7w2/?utm_source=share&utm_medium=web2x&context=3 my copy paste to
# SO: https://stackoverflow.com/a/75403918/1601580
tput rmcup

source $AFS/.bashrc.lfs

conda activate mds_env_gpu
#conda activate metalearning_gpu
export CUDA_VISIBLE_DEVICES=9
#export CUDA_VISIBLE_DEVICES=0,2
export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))")
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES; echo SLURM_JOBID = $SLURM_JOBID; echo hostname = $(hostname)
ulimit -n 120000; ulimit -Sn; ulimit -Hn

nvidia-smi
(echo "GPU_ID PID MEM% UTIL% UID APP" ; for GPU in 0 1 2 3 ; do for PID in $( nvidia-smi -q --id=${GPU} --display=PIDS | awk '/Process ID/{print $NF}') ; do echo -n "${GPU} ${PID} " ; nvidia-smi -q --id=${GPU} --display=UTILIZATION | grep -A4 -E '^[[:space:]]*Utilization' | awk 'NR=0{gut=0 ;mut=0} $1=="Gpu"{gut=$3} $1=="Memory"{mut=$3} END{printf "%s %s ",mut,gut}' ; ps -up ${PID} | gawk 'NR-1 {print $1,$NF}' ; done ; done) | column -t; hostname;
#nvidia-smi; (echo "GPU_ID PID UID APP" ; for GPU in 0 1 2 3 ; do for PID in $( nvidia-smi -q --id=${GPU} --display=PIDS | awk '/Process ID/{print $NF}') ; do echo -n "${GPU} ${PID} " ; ps -up ${PID} | awk 'NR-1 {print $1,$NF}' ; done ; done) | column -t; hostname; tmux ls;


(echo "GPU_ID PID UID APP" ; for GPU in 0 1 2 3 ; do for PID in $( nvidia-smi -q --id=${GPU} --display=PIDS | awk '/Process ID/{print $NF}') ; do echo -n "${GPU} ${PID} " ; ps -up ${PID} | awk 'NR-1 {print $1,$NF}' ; done ; done) | column -t

tmux new -s gpu0
tmux new -s gpu1
tmux new -s gpu2
tmux new -s gpu3
tmux new -s gpu4
tmux new -s gpu5
tmux new -s gpu6
tmux new -s gpu7
tmux new -s gpu8
tmux new -s gpu9

tmux new -s rand
tmux new -s rand0
tmux new -s rand1
tmux new -s rand2
tmux new -s rand3
tmux new -s rand4
tmux new -s rand5
tmux new -s rand6
tmux new -s rand7
tmux new -s rand8
tmux new -s rand9
tmux new -s rand10
tmux new -s rand11
tmux new -s rand12
tmux new -s rand13
tmux new -s rand14
tmux new -s rand15
tmux new -s rand16
tmux new -s rand17
tmux new -s rand18
tmux new -s rand19
tmux new -s rand20
tmux new -s rand21
tmux new -s rand22
tmux new -s rand23
tmux new -s rand24

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
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_hdb4_micod

# - mds maml
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_maml_torchmeta.py --manual_loads_name mds_resnet_maml_adam_scheduler --model_option resnet18_rfs --data_path $HOME/data/mds/records/
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_maml_torchmeta.py --manual_loads_name mds_resnet_maml_adam_scheduler --model_option resnet50_rfs --data_path $HOME/data/mds/records/
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_maml_torchmeta.py --manual_loads_name mds_maml --model_option resnet9_rfs --data_path $HOME/data/mds/records/
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_maml_torchmeta.py --manual_loads_name mds_maml --model_option resnet18_rfs --data_path $HOME/data/mds/records/
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_maml_torchmeta.py --manual_loads_name mds_maml --model_option resnet34_rfs --data_path $HOME/data/mds/records/
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_maml_torchmeta.py --manual_loads_name mds_maml --model_option resnet50_rfs --data_path $HOME/data/mds/records/

# - mds usl
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name mds_resnet_usl_adam_scheduler --model_option resnet18_rfs --data_path $HOME/data/mds/records/
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name mds_resnet_usl_adam_scheduler --model_option resnet50_rfs --data_path $HOME/data/mds/records/
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name mds_usl --model_option resnet9_rfs --data_path $HOME/data/mds/records/
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name mds_usl --model_option resnet18_rfs --data_path $HOME/data/mds/records/
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name mds_usl --model_option resnet34_rfs --data_path $HOME/data/mds/records/
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name mds_usl --model_option resnet50_rfs --data_path $HOME/data/mds/records/
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name mds_usl --model_option resnet50_rfs --data_path $HOME/data/mds/records/ --path_to_checkpoint "~/data/logs/logs_Feb03_15-02-19_jobid_231971_pid_1421508_wandb_True/ckpt.pt"

# - hdb4 micod usl
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_resnet_rfs_adam_cl_its --model_option resnet12_rfs
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_resnet_rfs_adam_cl_train_to_convergence --model_option resnet12_rfs
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_resnet_rfs_log_more_often_0p9_acc_reached --model_option resnet12_rfs

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_convg_reached_log_ckpt_more --filter_size 2 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_convg_reached_log_ckpt_more --filter_size 4 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_convg_reached_log_ckpt_more --filter_size 6 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_convg_reached_log_ckpt_more --filter_size 8 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_convg_reached_log_ckpt_more --filter_size 12 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_convg_reached_log_ckpt_more --filter_size 14 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_convg_reached_log_ckpt_more --filter_size 16 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_convg_reached_log_ckpt_more --filter_size 32 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_convg_reached_log_ckpt_more --filter_size 64 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_convg_reached_log_ckpt_more --filter_size 256 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_hdb4_micod_convg_reached_log_ckpt_more --filter_size 512 --model_option 5CNN_opt_as_model_for_few_shot

# - hdb4 micod maml
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod_resnet_rfs_scheduler_its --model_option resnet12_rfs
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod_resnet_rfs_scheduler_train_to_convergence --model_option resnet12_rfs

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod --filter_size 2 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod --filter_size 4 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod --filter_size 6 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod --filter_size 8 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod --filter_size 12 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod --filter_size 14 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod --filter_size 16 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod --filter_size 32 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod --filter_size 64 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod --filter_size 256 --model_option 5CNN_opt_as_model_for_few_shot
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod --filter_size 512 --model_option 5CNN_opt_as_model_for_few_shot

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_hdb4_micod --path_to_checkpoint "~/data/logs/logs_Feb02_14-00-49_jobid_991923_pid_2822438_wandb_True/ckpt.pt"



# -- mi 5cnns
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --filter_size 2 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --filter_size 6 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --filter_size 8 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --filter_size 16 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --filter_size 32 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --filter_size 64 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --filter_size 256 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --filter_size 512 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --filter_size 2 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --filter_size 6 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --filter_size 8 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --filter_size 16 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --filter_size 32 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --filter_size 64 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --filter_size 256 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --filter_size 512 --model_option 5CNN_opt_as_model_for_few_shot --data_option mini-imagenet





# -- mi, cifarfs; resnet12
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --model_option resnet12_rfs
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --model_option resnet12_rfs_cifarfs_fc100 --data_option cifarfs

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --model_option resnet12_rfs
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --model_option resnet12_rfs_cifarfs_fc100 --data_option cifarfs

# -- mi, hdb4; vit
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --model_option vit_mi --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --model_option vit_mi --data_option hdb4_micod
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --model_option vit_cifarfs --data_option cifarfs

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --model_option vit_mi --data_option mini-imagenet
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --model_option vit_mi --data_option hdb4_micod
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name maml_l2l --model_option vit_cifarfs --data_option cifarfs

# - mds vit (seperate, since it uses torchmeta & torchmeta doesn't seem to work with hf)
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name usl_l2l_data --model_option vit_mi --data_option mds

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_maml_torchmeta.py --manual_loads_name mds_maml --model_option vit_mi --data_option mds


# - performance comp usl vs maml
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_experiment_analysis_sl_vs_maml_performance_comp_distance.py --manual_loads_name resnet12rfs_hdb1_mio
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_experiment_analysis_sl_vs_maml_performance_comp_distance.py --manual_loads_name cifarfs_vit
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_experiment_analysis_sl_vs_maml_performance_comp_distance.py --manual_loads_name mi_vit
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_experiment_analysis_sl_vs_maml_performance_comp_distance.py --manual_loads_name resnet12rfs_cifarfs
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_experiment_analysis_sl_vs_maml_performance_comp_distance.py --manual_loads_name hdb4_micod

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_experiment_analysis_sl_vs_maml_performance_comp_distance.py --manual_loads_name mi
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_experiment_analysis_sl_vs_maml_performance_comp_distance.py --manual_loads_name hdb4_micod
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_experiment_analysis_sl_vs_maml_performance_comp_distance.py --manual_loads_name mds_full

# -- other option is to run `echo $SU_PASSWORD | /afs/cs/software/bin/reauth` inside of python, right?
export JOB_PID=$!
echo $OUT_FILE
echo $ERR_FILE
echo JOB_PID = $JOB_PID
echo SLURM_JOBID = $SLURM_JOBID

# - Done
echo "Done with the dispatching (daemon) sh script"
