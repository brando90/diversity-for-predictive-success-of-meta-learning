# -- send data from ampere4 to ampereX using rsync and creating dir if it doesn't exist
ssh brando9@ampere4.stanford.edu


# -- ampere 4 to ampere 1
tmux new -s mds2ampere2

source $AFS/.bashrc.lfs
conda activate mds_env_gpu
export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))")
export CUDA_VISIBLE_DEVICES=7; echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
ulimit -n 120000

# - send records
cd ~/data/mds
rsync -avW records brando9@ampere1:/lfs/local/0/brando9/data/mds
# - send everything from ampere4 to ampere1, ampere2, ampere3
# A trailing slash on a source path means "copy the contents of this directory". Without a trailing slash it means "copy the directory".
#cd ~/data/
ssh brando9@ampere4.stanford.edu
rsync -avW ~/data/mds brando9@ampere1:/lfs/local/0/brando9/data/mds
rsync -avW ~/data/mds brando9@ampere2:/lfs/local/0/brando9/data/mds
rsync -avW ~/data/mds brando9@ampere3:/lfs/local/0/brando9/data/mds
rsync -avW ~/data/mds brando9@hyperturing1:/lfs/local/0/brando9/data/mds
rsync -avW ~/data/mds brando9@hyperturing2:/lfs/local/0/brando9/data/mds
# fails, both can't be remote
# rsync -avW brando9@ampere4.stanford.edu:/lfs/ampere4/0/brando9/data/mds/ brando9@ampere1.stanford.edu:/lfs/ampere1/0/brando9/data/mds/

# -
tmux new -s mds2ampere1

source $AFS/.bashrc.lfs
conda activate mds_env_gpu
export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))")
export CUDA_VISIBLE_DEVICES=7; echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
ulimit -n 120000

cd ~/data/mds
rsync -avW records brando9@ampere2:/lfs/local/0/brando9/data/mds

#done
#cd ~/data/mds; rsync -avW records brando9@ampere3:/lfs/local/0/brando9/data/mds

# - send ckpts btw clusters
# send hyperturning2 to hyperturning1
# path1 usl: /lfs/hyperturing2/0/brando9/data/logs/logs_Mar16_18-01-56_jobid_478656_pid_136493_wandb_True
# there are already maml stuff there so only send path1 & instead path2 is some maml vit already in server
ssh brando9@hyperturing2.stanford.edu
# next command should show it's empty so that the content and folder (path) is sent
# cd /lfs/hyperturing1/0/brando9/data/logs/logs_Mar16_18-01-56_jobid_478656_pid_136493_wandb_True
#rsync -avW /lfs/hyperturing2/0/brando9/data/logs/logs_Mar16_18-01-56_jobid_478656_pid_136493_wandb_True brando9@hyperturing1:/lfs/hyperturing1/0/brando9/data/logs/logs_Mar16_18-01-56_jobid_478656_pid_136493_wandb_True





# - send l2l to ampere3
# todo idk if needed

