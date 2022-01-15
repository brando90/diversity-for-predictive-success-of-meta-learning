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
python -c "import meta_learning; print(meta_learning)"
python -c "import meta_learning; print(meta_learning); meta_learning.hello()"
```

output should be something like this:

```
(metalearning) brando~/automl-meta-learning/automl-proj-src ❯ python -c "import uutils; print(uutils); uutils.hello()"
<module 'uutils' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/__init__.py'>

hello from uutils __init__.py in:
<module 'uutils' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/__init__.py'>

(metalearning) brando~/automl-meta-learning/automl-proj-src ❯ python -c "import meta_learning; print(meta_learning)"
<module 'meta_learning' from '/Users/brando/automl-meta-learning/automl-proj-src/meta_learning/__init__.py'>
(metalearning) brando~/automl-meta-learning/automl-proj-src ❯ python -c "import meta_learning; print(meta_learning); meta_learning.hello()"
<module 'meta_learning' from '/Users/brando/automl-meta-learning/automl-proj-src/meta_learning/__init__.py'>

hello from torch_uu __init__.py in:
<module 'uutils.torch_uu' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/torch_uu/__init__.py'>
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