## Installation script

# - CAREFUL, if a job is already running it could do damage to it, rm reauth process, qian doesn't do it so skip it
# top -u brando9
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

# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
#bash ~/miniconda.sh -b -p $HOME/miniconda
#source ~/miniconda/bin/activate

# conda init zsh
conda init
conda install conda-build
conda update -n base -c defaults conda
conda update conda
conda update --all

pip install --upgrade pip
pip3 install --upgrade pip

#conda create -n metalearning_gpu python=3.9
#conda activate metalearning_gpu
## conda remove --name metalearning_gpu --all

pip install -U wandb

# MOVING THE TORCH INSTALL AFTER MY PACKAGES HAVE BEEN INSTALLED TO FORCE THE RIGHT PYTORCH VERSION & IT's CUDA TOOLKIT
# DO NOT CHANGE THIS, this is needed for the vision cluster & the librarires we are using.
# Only works with python 3.9
# todo - would be nice to have an if statement if we are in the vision cluster...extra work, probably not worth it
# for now the hack is to try cuda, then if fail try normal, to make sure that didn't overwrite the prev if cuda
# succeeded try again, if it fails it won't do anything.
# Overall install cuda if gpu available o.w. it install the normal version.
# any other behaviour is likely unexpected.
#pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#pip3 install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

#pip3 install torch==1.13.0+cu111 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
# conda install -y -c pytorch -c conda-forge cudatoolkit=11.1 pytorch torchvision torchaudio

#host_v=$(hostname)
#if [ $host_v = vision-submit.cs.illinois.edu ]; then
##    echo "Strings are equal."
#  pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#else
##    echo "Strings are not equal."
#  pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#fi

# not sure if needed but leaving here for now
# conda install -y pyyml
# pip install pyyml
# pip install learn2learn

git clone git@github.com:brando90/ultimate-utils.git
git clone git@github.com:brando90/diversity-for-predictive-success-of-meta-learning.git
#git clone git@github.com:brando90/ultimate-anatome.git
#git clone git@github.com:brando90/ultimate-aws-cv-task2vec.git
#git clone git@github.com:brando90/pycoq.git
#git clone git@github.com:FormalML/iit-term-synthesis.git

pip install -e ~/ultimate-utils/
pip install -e ~/diversity-for-predictive-success-of-meta-learning/
#pip install -e ~/ultimate-anatome/
#pip install -e ~/ultimate-aws-cv-task2vec/

# - todo: test, decided to use conda only for pytorch since cudatoolkit is easier to specify & get the most recent torch version
source cuda11.1
# To see Cuda version in use
nvcc -V
# torchmeta needs pytorch < 1.10.0

# - pytorch install
#conda uninstall pytorch
#pip uninstall torch
# official: https://pytorch.org/get-started/previous-versions/
#pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#pip3 install torch==1.13.0+cu111 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
#conda install -y -c pytorch -c conda-forge cudatoolkit=11.1 pytorch torchvision torchaudio
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 --upgrade
python -c "import torch; print(torch.__version__)"
python -c "import uutils; uutils.torch_uu.gpu_test()"

echo Done "Done Install!"

# -- extra notes
# - using conda develop rather than pip because uutils installs incompatible versions with the vision cluster
## python -c "import sys; [print(p) for p in sys.path]"
#conda install conda-build
#conda develop ~/ultimate-utils/ultimate-utils-proj-div_src
#conda develop -u ~/ultimate-utils/ultimate-utils-proj-div_src
