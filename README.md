#diversity-for-predictive-success-of-meta-learning

# Installing

Installing software can be tricky so follow the instructions carefully (and skeptically).

## Manual installation [Recommended, Development]

To use library first gitclone it as you'd usually do.

Activate your virtual env and make sure it uses **python 3.9** 
or create one as follows:

```
conda create -n meta_learning python=3.9
conda activate meta_learning
```
 
Then you will install the rest by running 
(please read the script before you run it blindly, have an idea of what it is doing):
```angular2html
./install.sh
```

You might have to do `chmod a+x install.sh` for it to run from your terminal
to give it the right permissions. 

### Test installation

To test that uutils install do:
```
python -c "import uutils; print(uutils); uutils.hello()"
```
output should be something like this:
```
(meta_learning_a100) [miranda9@hal-dgx diversity-for-predictive-success-of-meta-learning]$ python -c "import uutils; print(uutils); uutils.hello()"
<module 'uutils' from '/home/miranda9/ultimate-utils/ultimate-utils-proj-src/uutils/__init__.py'>

hello from uutils __init__.py in:
<module 'uutils' from '/home/miranda9/ultimate-utils/ultimate-utils-proj-src/uutils/__init__.py'>
```


GPU TEST: To test if pytorch works with gpu do (it should fail if no gpus are available):
```
python -c "import uutils; uutils.torch_uu.gpu_test()"
```
output should be something like this:
```
(meta_learning_a100) [miranda9@hal-dgx diversity-for-predictive-success-of-meta-learning]$ python -c "import uutils; uutils.torch_uu.gpu_test()"
device name: A100-SXM4-40GB
Success, no Cuda errors means it worked see:
out=tensor([[ 0.5877],
        [-3.0269]], device='cuda:0')
```

### Appendix for Installation

Follow the above instructions carefully and use the vision cluster.
Installation can be tricky, time intensive and hardware details can make it even harder.
This has not been tested in HAL and using HAL is not encouraged since it might be hard to
set up and this already works on the vision cluster.


# Running experiments

To run experiments use the standard vision cluster (condor) way to run things 
beyond interactive jobs (learn how to do that).

The main things to have at the top of your script is:
```angular2html
# -- setup up for condor_submit background script in vision-cluster
export HOME=/home/miranda9
# to have modules work and the conda command work
source /etc/bashrc
source /etc/profile
source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile

module load gcc/9.2.0
#module load cuda-toolkit/10.2
module load cuda-toolkit/11.1
```

to make sure your main script is run

# Data storage at UIUC vision cluster IC

My goal is to put the large heavy stuff (e.g. conda, data, ) at `/shared/rsaas/miranda9/`.
Warning: due to the vpn if you run one of this commands and you lose connection you will have to do it again and might
have half a transfer of files. 
So run them in a job.sub command or re pull them from git then do a soft link.
```bash
mv ~/data /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/data ~/data 

mv ~/miniconda3 /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/miniconda3 ~/miniconda3

mv ~/data_folder_fall2020_spring2021 /shared/rsaas/miranda9/
ln -s /shared/rsaas/miranda9/data_folder_fall2020_spring2021 ~/data_folder_fall2020_spring2021


mv ~/diversity-for-predictive-success-of-meta-learning /shared/rsaas/miranda9
git
ln -s /shared/rsaas/miranda9/diversity-for-predictive-success-of-meta-learning ~/diversity-for-predictive-success-of-meta-learning 

mv ~/Does-MAML-Only-Work-via-Feature-Re-use-A-Data-Set-Centric-Perspective /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/Does-MAML-Only-Work-via-Feature-Re-use-A-Data-Set-Centric-Perspective ~/Does-MAML-Only-Work-via-Feature-Re-use-A-Data-Set-Centric-Perspective 

mv ~/ultimate-anatome /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/ultimate-anatome ~/ultimate-anatome 

mv ~/ultimate-aws-cv-task2vec /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/ultimate-aws-cv-task2vec ~/ultimate-aws-cv-task2vec 

mv ~/ultimate-utils /shared/rsaas/miranda9
git 
ln -s /shared/rsaas/miranda9/ultimate-utils ~/ultimate-utils 
```

# Data

For the SL experiments we used the rfs repo's data: https://github.com/WangYueFt/rfs
For the meta-learning experiment we used:
- torchmeta's data: https://github.com/tristandeleu/pytorch-meta
- TODO: hope to use learn2learn for distriuted meta-training: https://learn2learn.net/

# Contributions

Place reusable code in src in the appropriate place (use your judgement).
I suggest not to only rely on Jupyter and use real python scripts - especially for code
that should be reusable and tested.
The folder results_plots is mainly intended for scripts to create plots 
for Papers.
I will put my python main experiments in `experiment_mains` and run it from a
`main.sh` file that is submitted through the `job.sub` file.
Make sure to give the `main.sh` the right permissions for condor to work
e.g. `chmod a+x ~/automl-meta-learning/main.sh`.
An example submission job would be:
```
chmod a+x ~/automl-meta-learning/main.sh
condor_submit job.sub
```
then you can track the experiment live with:
```angular2html
tail -f main.sh.oXXXX
```
if you are basing your experiments on my files.

I suggest to read:
- main.sh
- main.py
- job.sub
- interactive.sub

to have an idea how to run jobs.
For interactive jobs I do `condor -i interactive.sub`.
For papers results all experiments need to be recorded in wandb so learn how to use it.
Including number you produce for external figures.

Good luck and have fun! Stay curious. Onwards!