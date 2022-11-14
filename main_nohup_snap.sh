# https://unix.stackexchange.com/questions/724902/how-does-one-send-new-commands-to-run-to-an-already-running-nohup-process-e-g-r
# sh ~/diversity-for-predictive-success-of-meta-learning/main_nohup_snap.sh
# - set up this main sh script
source ~/.bashrc
source ~/.bash_profile
source ~/.bashrc.user
echo HOME = $HOME

source cuda11.1

conda init bash
conda activate metalearning_gpu

# - get a job id for this tmux session
export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))")
echo SLURM_JOBID = $SLURM_JOBID
export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
export ERR_FILE=$PWD/main.sh.e$SLURM_JOBID
export WANDB_DIR=$HOME/wandb_dir

export CUDA_VISIBLE_DEVICES=4
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
python -c "import torch; print(torch.cuda.get_device_name(0));"

# - CAREFUL, if a job is already running it could do damage to it, rm reauth process, qian doesn't do it so skip it
# top -u brando9
# pkill -9 reauth -u brando9

# - expt python script then inside that python pid attach a reauth process
# should I run rauth within python with subprocess or package both the nohup command and the rauth together in badsh somehow
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_hdb1_5cnn_adam_cl_filter_size --filter_size 4 > $OUT_FILE 2> $ERR_FILE &

#nohup sh -c 'echo $SU_PASSWORD | /afs/cs/software/bin/reauth; python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_hdb1_5cnn_adam_cl_filter_size --filter_size 4 > $OUT_FILE 2> $ERR_FILE' > nohup.out$SLURM_JOBID &

nohup python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_hdb1_5cnn_adam_cl_filter_size --filter_size 4 > $OUT_FILE 2> $ERR_FILE &

# other option is to run `echo $SU_PASSWORD | /afs/cs/software/bin/reauth` inside of python, right?
export JOB_PID=$!
echo JOB_PID = $JOB_PID

# - Done
echo "Done with the dispatching (daemon) sh script