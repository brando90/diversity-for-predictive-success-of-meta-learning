## Installation script

#cd $HOME/diversity-for-predictive-success-of-meta-learning/
cd $HOME/
echo $HOME

# - install miniconda locally
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

conda create -n mds_env_gpu python=3.9
conda activate mds_env_gpu
## conda remove --name mds_env_gpu --all
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r $HOME/diversity-for-predictive-success-of-meta-learning/req_mds_essentials.txt

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
# NOTE: YOU **NECESSERILY** NEED TO INSTALL CPU or GPU MANUALLY (can't be put in setup.py afaik, I guess it can, its turing complete. Would go in div setup.py)
pip uninstall torchtext
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# - bellow doesn't work, idk why. ref SO Q: https://stackoverflow.com/questions/75023120/why-does-conda-install-the-pytorch-cpu-version-despite-me-putting-explicitly-to
#conda install pytorch torchvision torchaudio pytorch-cuda=11.1 -c pytorch -c nvidia
#conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# - test pytorch with cuda
python -c "import torch; print(torch.__version__); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"

# - git requirements
cd $HOME
git clone git@github.com:brando90/diversity-for-predictive-success-of-meta-learning.git
pip install -e $HOME/diversity-for-predictive-success-of-meta-learning/

#pip install statsmodels  # not sure why it's needed hardcoded and in setup.py uutils is not enough
git clone git@github.com:brando90/ultimate-utils.git
pip install -e $HOME/ultimate-utils/

# temporary, should be in setup.py for uutils and reqs mds .txt file
pip install fairseq
#pip install setuptools==59.5.0
# not sure why it's needed hardcoded and in setup.py uutils is not enough
pip install statsmodels==0.13.5

# - test uutils was installed and gpus are working
python -c "import torch; print(torch.__version__)"
python -c "import uutils; uutils.torch_uu.gpu_test()"

echo Done "Done Install!"
