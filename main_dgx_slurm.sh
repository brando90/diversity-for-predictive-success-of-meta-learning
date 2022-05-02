#!/bin/bash
#SBATCH --job-name="brando_job"
#SBATCH --output="main.sh.o%j.%N"
#SBATCH --partition=x86
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=16
#SBATCH --threads-per-core=2
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:a100:1
#SBATCH --export=ALL

# -- Example 1
# Request 2x GPU along with 64x CPU cores for 12 hours
# srun --partition=x86 --time=12:00:00 --nodes=1 --ntasks-per-node=64 --sockets-per-node=1 --cores-per-socket=32
# --threads-per-core=2 --mem-per-cpu=4000 --wait=0 --export=ALL --gres=gpu:a100:2 --pty /bin/bash

# -- Example 2
# Request 4x GPU along with 128x CPU cores for 24 hours
# srun --partition=x86 --time=24:00:00 --nodes=1 --ntasks-per-node=128 --sockets-per-node=1 --cores-per-socket=64
# --threads-per-core=2 --mem-per-cpu=4000 --wait=0 --export=ALL --gres=gpu:a100:4 --pty /bin/bash

# - Print stuff
cd ~
echo STARTING `date`
# why srun inside sbatch: https://stackoverflow.com/questions/53636752/slurm-why-use-srun-inside-sbatch
srun hostname

echo CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
echo torch.cuda.device_count is:
python -c "import torch; print(torch.cuda.device_count())"
echo ---- Running your python main ----

/home/miranda9/miniconda3/envs/meta_learning_a100/bin/python -m pip install --upgrade pip
pip install wandb --upgrade

# - SL
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_32_filter_size

# - MAML
#python -m torch.distributed.run --nproc_per_node=3 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_mi_adam_filter_size
#python -m torch.distributed.run --nproc_per_node=3 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_mi_adam_filter_size_32_filter_size

# - Data analysis
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main2_distance_sl_vs_maml.py
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/_main_distance_sl_vs_maml.py

echo -- Done submitting job in dgx A100-SXM4-40G