## Installation script

#cd $HOME/diversity-for-predictive-success-of-meta-learning/
cd $HOME/
echo $HOME

# - install mini conda
##wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash $HOME/miniconda.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate

# - installing full anaconda
#wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/anaconda.sh
#wget https://repo.continuum.io/conda/Anaconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh
#nohup bash ~/anaconda.sh -b -p $HOME/anaconda > anaconda_install.out &
#ls -lah $HOME | grep anaconda
#source ~/anaconda/bin/activate

# - set up conda
conda init
# conda init zsh
conda install conda-build
conda update -n base -c defaults conda
conda update conda
conda update --all

# - create conda env
conda create -n metalearning_gpu python=3.9
conda activate metalearning_gpu
## conda remove --name metalearning_gpu --all

# - make sure pip is up to date
which python
pip install --upgrade pip
pip3 install --upgrade pip
which pip
which pip3

# -- Install PyTorch sometimes requires more careful versioning due to cuda, ref: official install instruction https://pytorch.org/get-started/previous-versions/
# you need python 3.9 for torch version 1.9.1 to work, due to torchmeta==1.8.0 requirement
if ! python -V 2>&1 | grep -q 'Python 3\.9'; then
    echo "Error: Python 3.9 is required!"
    exit 1
fi
# - install torch 1.9.1 with cuda using pip
pip uninstall torchtext
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# - bellow doesn't work, idk why. ref SO Q: https://stackoverflow.com/questions/75023120/why-does-conda-install-the-pytorch-cpu-version-despite-me-putting-explicitly-to
#conda install pytorch torchvision torchaudio pytorch-cuda=11.1 -c pytorch -c nvidia
#conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# - test pytorch with cuda
python -c "import torch; print(torch.__version__); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"

# - git requirements
cd $HOME
git clone git@github.com:brando90/diversity-for-predictive-success-of-meta-learning.git
pip install -e $HOME/diversity-for-predictive-success-of-meta-learning/

git clone git@github.com:brando90/ultimate-utils.git
pip install -e $HOME/ultimate-utils/

# - test uutils was installed and gpus are working
python -c "import torch; print(torch.__version__)"
python -c "import uutils; uutils.torch_uu.gpu_test()"

##git clone -b hdb git@github.com:brando90/meta-dataset.git
##pip install -e -r $HOME/meta-dataset/requirements.txt
#
##git clone -b hdb git@github.com:brando90/pytorch-meta-dataset.git
##pip install -e -r $HOME/pytorch-meta-dataset/requirements.txt
#
## - git submodule install
#cd $HOME/diversity-for-predictive-success-of-meta-learning
#
## - in case it's needed if the submodules bellow have branches your local project doesn't know about from the submodules upstream
#git fetch
#
## - adds the repo to the .gitmodule & clones the repo
#git submodule add -f -b hdb --name meta-dataset git@github.com:brando90/meta-dataset.git meta-dataset/
#git submodule add -f -b hdb --name pytorch-meta-dataset git@github.com:brando90/pytorch-meta-dataset.git pytorch-meta-dataset/
#
## - git submodule init initializes your local configuration file to track the submodules your repository uses, it just sets up the configuration so that you can use the git submodule update command to clone and update the submodules.
#git submodule init
## - The --remote option tells Git to update the submodule to the commit specified in the upstream repository, rather than the commit specified in the main repository. ref: https://stackoverflow.com/questions/74988223/why-do-i-need-to-add-the-remote-to-gits-submodule-when-i-specify-the-branch?noredirect=1&lq=1
#git submodule update --init --recursive --remote
#
## - for each submodule pull from the right branch according to .gitmodule file. ref: https://stackoverflow.com/questions/74988223/why-do-i-need-to-add-the-remote-to-gits-submodule-when-i-specify-the-branch?noredirect=1&lq=1
##git submodule foreach -q --recursive 'git switch $(git config -f $toplevel/.gitmodules submodule.$name.branch || echo master || echo main )'
#
## - check it's in specified branch. ref: https://stackoverflow.com/questions/74998463/why-does-git-submodule-status-not-match-the-output-of-git-branch-of-my-submodule
#git submodule status
#cd meta-dataset
#git branch
#cd ..
#
## - pip install. ref: https://stackoverflow.com/questions/75010219/how-do-i-pip-install-something-in-editable-mode-using-a-requirements-txt-file/75010220#75010220
#pip install --upgrade pip
##pip install -r meta-dataset/requirements.txt -e meta-dataset
##pip install -r pytorch-meta-dataset/requirements.txt -e pytorch-meta-dataset
## don't think the requirements.txt file is needed, the setup.py has the same stuff
#pip install -e meta-dataset
#pip install -e pytorch-meta-dataset

# - create conda script for mds
conda update -n base -c defaults conda

conda create -n mds_env_gpu python=3.9
conda activate mds_env_gpu
pip install -r $HOME/diversity-for-predictive-success-of-meta-learning/requirements_patrick_mds.txt

cd $HOME/diversity-for-predictive-success-of-meta-learning

#
#git clone git@github.com:brando90/pytorch-meta-dataset.git $HOME/pytorch-meta-dataset
#cd $HOME/pytorch-meta-dataset
#git branch hdb c6d6922003380342ab2e3509425d96307aa925c5
#git checkout hdb
#git push -u origin hdb

# - todo: test, decided to use conda only for pytorch since cudatoolkit is easier to specify & get the most recent torch version
source cuda11.1
# To see Cuda version in use
nvcc -V
# torchmeta needs pytorch < 1.10.0



echo Done "Done Install!"

# -- extra notes
# - using conda develop rather than pip because uutils installs incompatible versions with the vision cluster
## python -c "import sys; [print(p) for p in sys.path]"
#conda install conda-build
#conda develop ~/ultimate-utils/ultimate-utils-proj-div_src
#conda develop -u ~/ultimate-utils/ultimate-utils-proj-div_src
