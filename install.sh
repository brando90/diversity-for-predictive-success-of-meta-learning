## Installation script
# to install do: bash ~/automl-meta-learning/install.sh

conda update conda
pip install --upgrade pip
pip3 install --upgrade pip

# DO NOT CHANGE THIS, this is needed for the vision cluster & the librarires we are using.
# Only works with python 3.9
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# not sure if needed but leaving here for now
# conda install -y pyyml
# pip install pyyml
# pip install learn2learn

pip install -e ~/ultimate-utils/
pip install -e ~/ultimate-anatome/
pip install -e ~/ultimate-aws-cv-task2vec/
pip install -e ~/dimo-differentiable-model-optimization/

# -- extra notes

# - using conda develop rather than pip because uutils installs incompatible versions with the vision cluster
## python -c "import sys; [print(p) for p in sys.path]"
conda install conda-build
conda develop ~/ultimate-utils/ultimate-utils-proj-src
conda develop ~/automl-meta-learning/automl-proj-src
