# Running last time mi worked

```bash
git checkout -b <new_branch_name> <sha1>

#div
git checkout -b last_time_mi_worked bfb367ace63d2210f17082c7c4eed8e2a1e7dc36 
#uutils
git checkout -b last_time_mi_worked 3e4932203fc1e3cb2ea87b49b2069e1d9b0a760b 

git checkout <branch name> -- <path/to/file>
git checkout last_time_mi_worked -- 

# If you want to restore all the missing files from the local repository https://stackoverflow.com/questions/9705626/how-do-i-pull-a-missing-file-back-into-my-branch
git checkout last_time_mi_worked .
```

Experiment commands
```bash
python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_mi_rfs_adam_cl_100k

python -m torch.distributed.run --nproc_per_node=1 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_mi_rfs_adam_cl_100k

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_mi_rfs_adam_cl_100k
```

Edits
```bash
vim ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py
```

Both libs need to math:
```bash
#uutils
3e4932203fc1e3cb2ea87b49b2069e1d9b0a760b

#div
bfb367ace63d2210f17082c7c4eed8e2a1e7dc36
```

```bash
Request_gpus = 1
# requirements = (CUDADeviceName != "Tesla K40m")
# requirements = (CUDADeviceName == "Quadro RTX 6000")
# requirements = (CUDADeviceName == "Titan Xp")
# requirements = (CUDADeviceName == "NVIDIA A40") || (CUDADeviceName == "Quadro RTX 6000")
requirements = (CUDADeviceName == "NVIDIA A40")

# Request_cpus = 4
Request_cpus = 8
# Request_cpus = 16
# Request_cpus = 32
# Request_cpus = 40
# Request_cpus = 12
# Request_cpus = 32
Queue
```

# Running before hdb1

```bash
git checkout -b <new_branch_name> <sha1>

# activate submission file
chmod a+x ~/diversity-for-predictive-success-of-meta-learning/main.sh

# did a git stash on div for job.sub
git stash

#div
git checkout -b before_hdb1 a06f98b0deab4ac27cfc849636ea57e4a0053d58
#uutils
git checkout -b before_hdb1 1c9d4f4e235501ee27a5c3445686ab5a1d052e89 

# If you want to restore all the missing files from the local repository https://stackoverflow.com/questions/9705626/how-do-i-pull-a-missing-file-back-into-my-branch
git checkout last_time_mi_worked .

condor_submit ~/diversity-for-predictive-success-of-meta-learning/job.sub
```

```bash
cat ~/diversity-for-predictive-success-of-meta-learning/job.sub
```