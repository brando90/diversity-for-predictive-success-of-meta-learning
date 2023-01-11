# based on: https://github.com/google-research/meta-dataset/blob/main/doc/dataset_conversion.md
# make index is needed, it's at the last step at the end of this doc (after installing all the tfrecords from MDS)
# Q: what is
#   --splits_root=$SPLITS \
#   --records_root=$RECORD
# SPLITS is what does the hierarchical splitting of classes when sampling tasks (you should have a few json files),
# RECORDS is where the tfrecords are recorded. plus a json file dataspec to map classes to json files.

# - you might need to kill your reauths if your starting from scratch
# pkill -9 reauth -u brando9;

# - if your doing this in snap you might need to do
ssh brando9@ampere4.stanford.edu
conda activate metalearning_gpu

krbtmux
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

# perhaps attach to an running session
tmux attach -t <session number>

# -- prereqs: install gsutil
# if that doesnt work here's googles OS-specific instructions to install gsutil: https://download.huihoo.com/google/gdgdevkit/DVD1/developers.google.com/storage/docs/gsutil_install.html
# see "Installing from the Python package index (PyPi)"
pip install gsutil
# -- Option1: git clone in $HOME and pip install -e (if not already done)
# -- prereqs: download pytorch-mds to $HOME
cd $HOME
git clone https://github.com/brando90/pytorch-meta-dataset # done already?
# -- prereqs: download original mds to $HOME
git clone https://github.com/google-research/meta-dataset
#pip install -r meta-dataset/requirements.txt
#pip install -r pytorch-meta-dataset/requirements.txt
pip install -r $HOME/diversity-for-predictive-success-of-meta-learning/req_mds_essentials.txt

# -- Option2: using gitsubmodules.
# - git submodule install
cd $HOME/diversity-for-predictive-success-of-meta-learning
# - in case it's needed if the submodules bellow have branches your local project doesn't know about from the submodules upstream
git fetch
# - adds the repo to the .gitmodule & clones the repo
git submodule add -f -b hdb --name meta-dataset git@github.com:brando90/meta-dataset.git meta-dataset/
git submodule add -f -b hdb --name pytorch-meta-dataset git@github.com:brando90/pytorch-meta-dataset.git pytorch-meta-dataset/

# - git submodule init initializes your local configuration file to track the submodules your repository uses, it just sets up the configuration so that you can use the git submodule update command to clone and update the submodules.
git submodule init
# - The --remote option tells Git to update the submodule to the commit specified in the upstream repository, rather than the commit specified in the main repository. ref: https://stackoverflow.com/questions/74988223/why-do-i-need-to-add-the-remote-to-gits-submodule-when-i-specify-the-branch?noredirect=1&lq=1
git submodule update --init --recursive --remote
# - for each submodule pull from the right branch according to .gitmodule file. ref: https://stackoverflow.com/questions/74988223/why-do-i-need-to-add-the-remote-to-gits-submodule-when-i-specify-the-branch?noredirect=1&lq=1
#git submodule foreach -q --recursive 'git switch $(git config -f $toplevel/.gitmodules submodule.$name.branch || echo master || echo main )'
# - check it's in specified branch. ref: https://stackoverflow.com/questions/74998463/why-does-git-submodule-status-not-match-the-output-of-git-branch-of-my-submodule
git submodule status
cd meta-dataset
git branch
cd ..
# - pip install mds in this exact order
pip install -e $HOME/diversity-for-predictive-success-of-meta-learning/meta-dataset/
pip install -e $HOME/diversity-for-predictive-success-of-meta-learning/pytorch-meta-dataset/
pip install -r $HOME/diversity-for-predictive-success-of-meta-learning/req_mds_essentials.txt

# - install other stuff in case they are missing
if ! python -V 2>&1 | grep -q 'Python 3\.9'; then
    echo "Error: Python 3.9 is required!"
    exit 1
fi
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e $HOME/diversity-for-predictive-success-of-meta-learning/
pip install -e $HOME/ultimate-utils/

python -c "import torch; print(torch.__version__); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"
python -c "import uutils; uutils.torch_uu.gpu_test()"

# -- create records and splits folders for mds
echo $HOME
mkdir -p $HOME/data/mds
export MDS_DATA_PATH=$HOME/data/mds

mkdir -p $MDS_DATA_PATH/records
mkdir -p $MDS_DATA_PATH/splits
export RECORDS=$MDS_DATA_PATH/records
export SPLITS=$MDS_DATA_PATH/splits

echo $MDS_DATA_PATH
echo $RECORDS
echo $SPLITS

# in order to run python scripts we need to cd into the original mds dir
cd $HOME/meta-dataset/
# or
cd $HOME/diversity-for-predictive-success-of-meta-learning/meta-dataset/

# -- ilsvrc_2012: https://github.com/google-research/meta-dataset/blob/main/doc/dataset_conversion.md#ilsvrc_2012
ssh brando9@ampere4.stanford.edu
tmux new -s ilsvrc_2012
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

tmux attach -t ilsvrc_2012

## - 1. Download ilsvrc2012_img_train.tar, from the ILSVRC2012 website (about 84m-91m +- a lot)
## todo: https://gist.github.com/bonlime/4e0d236cf98cd5b15d977dfa03a63643
## todo: https://github.com/google-research/meta-dataset/blob/main/doc/dataset_conversion.md#ilsvrc_2012
## wget TODO -O $MDS_DATA_PATH/ilsvrc_2012
## for imagenet url: https://image-net.org/download-images.php
#wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -O $HOME/data/ILSVRC2012_img_train.tar
## 18GB
#ls -lh $HOME/data/ILSVRC2012_img_train.tar
#
## - 2. Extract it into ILSVRC2012_img_train/, which should contain 1000 files, named n????????.tar (expected time: ~30 minutes) ref: https://superuser.com/questions/348205/how-do-i-unzip-a-tar-gz-archive-to-a-specific-destination
#mkdir -p $HOME/data/ILSVRC2012_img_train
#tar xf $HOME/data/ILSVRC2012_img_train.tar -C $HOME/data/ILSVRC2012_img_train
## expected time: ~30 minutes & should contain 1000 files, named n????????.tar
#ls $HOME/data/ILSVRC2012_img_train | grep -c .tar
##ls $HOME/data/ | grep -c .tar
## count the number of .tar files in current dir (doesn't not work recursively, for that use find)
#if [ $(ls $HOME/data/ILSVRC2012_img_train | grep -c "\.tar$") -ne 1000 ]; then
#  echo "Error: expected 1000 .tar files, found $(ls | grep -c "\.tar$")"
##  exit 1
#else
#  echo "Success"
#fi
## to finish extracting into ILSVRC2012_img_train/ you need to move the files
#mkdir -p $MDS_DATA_PATH/ILSVRC2012_img_train/
#mv $HOME/data/ILSVRC2012_img_train/* $MDS_DATA_PATH/ILSVRC2012_img_train/
## check files are there
#ls $MDS_DATA_PATH/ILSVRC2012_img_train/
#ls $MDS_DATA_PATH/ILSVRC2012_img_train/ | grep -c .tar
## should still be 1000
#if [ $(ls $MDS_DATA_PATH/ILSVRC2012_img_train | grep -c "\.tar$") -ne 1000 ]; then
#  echo "Error: expected 1000 .tar files, found $(ls | grep -c "\.tar$")"
##  exit 1
#else
#  echo "Success"
#fi
#
## - 3. Extract each of ILSVRC2012_img_train/n????????.tar in its own directory (expected time: ~30 minutes), for instance:
#ls $MDS_DATA_PATH/ILSVRC2012_img_train/ | grep -c .tar
#ls -lh $MDS_DATA_PATH/ILSVRC2012_img_train
#for FILE in $MDS_DATA_PATH/ILSVRC2012_img_train/*.tar;
#do
#  echo ---
#  echo $FILE
#  # remove . tar from the end so create a dir of name FILE
#  mkdir -p ${FILE/.tar/};
#  cd ${FILE/.tar/};
##  tar xvf ../$FILE;
#  tar xvf $FILE -C ${FILE/.tar/};
#  cd ..;
#done
##
#ls ${FILE/.tar/} | grep -c .JPEG
## 1300
## (expected time: ~30 minutes)
#ls $MDS_DATA_PATH/ILSVRC2012_img_train/
#ls $MDS_DATA_PATH/ILSVRC2012_img_train/ | grep -c .tar
#ls $MDS_DATA_PATH/ILSVRC2012_img_train/ -1 | grep -v "\.tar$" | wc -l
#
## - 4. Download the following two files into ILSVRC2012_img_train/
#wget http://www.image-net.org/data/wordnet.is_a.txt -O $MDS_DATA_PATH/ILSVRC2012_img_train/wordnet.is_a.txt
#wget http://www.image-net.org/data/words.txt -O $MDS_DATA_PATH/ILSVRC2012_img_train/words.txt
##
#cat $MDS_DATA_PATH/ILSVRC2012_img_train/wordnet.is_a.txt
#cat $MDS_DATA_PATH/ILSVRC2012_img_train/words.txt
#ls $MDS_DATA_PATH/ILSVRC2012_img_train/ | grep -c "*"
#
## - 5. Launch the conversion script (Use --dataset=ilsvrc_2012_v2 for the training only MetaDataset-v2 version):
#python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
#  --dataset=ilsvrc_2012 \
#  --ilsvrc_2012_data_root=$MDS_DATA_PATH/ILSVRC2012_img_train \
#  --splits_root=$SPLITS \
#  --records_root=$RECORDS
#
## -6. Expect the conversion to take 4 to 12 hours, depending on the filesystem's latency and bandwidth.

# --ilsvrc, shortcut since original install didnt succceed
ssh brando9@ampere4.stanford.edu
tmux new -s ilsvrc_2012_gdown
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

tmux attach -t ilsvrc_2012
#option 1. assuming you still have uiuc netid and can access the /shared/rsaas folder
#scp <your net id>@vision.cs.illinois.edu:/shared/rsaas/pzy2/ilsvrc.tar.gz <wherever your $RECORDS destination at your stanford cluster>

#option 2. not sure if it works since I got this issue when i tried: https://stackoverflow.com/questions/65312867/how-to-download-large-file-from-google-drive-from-terminal-gdown-doesnt-work
# tool to "wget" gdrive links
pip install gdown
# actual file here https://drive.google.com/file/d/1FkuIIW49_x8Zzr6Vmpq-mHLO69-VvCtX/view?usp=share_link
gdown https://drive.google.com/uc?id=1FkuIIW49_x8Zzr6Vmpq-mHLO69-VvCtX


#check that the md5hash matches my (working) tar.gz file
md5sum $RECORDS/ilsvrc.tar.gz #should be 56c576d10896bfa8d35200aebfea1704

#should have ilsvrc.tar.gz in $RECORDS/
#now extract it
tar -xf $RECORDS/ilsvrc.tar.gz -C $RECORDS/

# Need to un-nest folders since I extracted my mscoco at the top-most directory instead of in $RECORDS/
mv $RECORDS/shared/rsaas/pzy2/records/ilsvrc_2012  $RECORDS/ilsvrc_2012


# ilsvrc doesn't have a dedicated splits file, so should be done
# -7.Find the following outputs in $RECORDS/ilsvrc_2012/:
#1000 tfrecords files named [0-999].tfrecords
ls $RECORDS/ilsvrc_2012/ | grep -c .tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/ilsvrc_2012/dataset_spec.json
#num_leaf_images.json
ls $RECORDS/ilsvrc_2012/num_leaf_images.json

# -7.Find the following outputs in $RECORDS/ilsvrc_2012/:
#1000 tfrecords files named [0-999].tfrecords
ls $RECORDS/ilsvrc_2012/ | grep -c .tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/ilsvrc_2012/dataset_spec.json
#num_leaf_images.json
ls $RECORDS/ilsvrc_2012/num_leaf_images.json



# -- omniglot: https://github.com/google-research/meta-dataset/blob/main/doc/dataset_conversion.md#omniglot
# if your on snap you might want to create a new tmux session
ssh brando9@ampere4.stanford.edu
tmux new -s omniglot
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

# tmux attach -t omniglot

# 1. Download images_background.zip and images_evaluation.zip
mkdir -p $MDS_DATA_PATH/omniglot/
wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip -O $MDS_DATA_PATH/omniglot/images_background.zip
wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip -O $MDS_DATA_PATH/omniglot/images_evaluation.zip

# 2. Extract them into the same omniglot/ directory
unzip $MDS_DATA_PATH/omniglot/images_background.zip -d $MDS_DATA_PATH/omniglot
unzip $MDS_DATA_PATH/omniglot/images_evaluation.zip -d $MDS_DATA_PATH/omniglot

ls $MDS_DATA_PATH/omniglot

# 3. Launch the conversion script:
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=omniglot \
  --omniglot_data_root=$MDS_DATA_PATH/omniglot \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

# 3. Expect the conversion to take a few seconds.

# 4. Find the following outputs in $RECORDS/omniglot/:
#1623 tfrecords files named [0-1622].tfrecords
ls $RECORDS/omniglot/ | grep -c .tfrecords
#find $RECORDS/omniglot/ -name "*.tfrecords" | wc -l
#dataset_spec.json (see note 1)
cat $RECORDS/omniglot/dataset_spec.json




# -- aircraft: https://github.com/google-research/meta-dataset/blob/main/doc/dataset_conversion.md#aircraft
ssh brando9@ampere4.stanford.edu
tmux new -s aircraft
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

#tmux attach -t aircraft

# 1. download and extract
wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz -O $MDS_DATA_PATH/fgvc-aircraft-2013b.tar.gz

# 2. Extract it into fgvc-aircraft-2013b
tar xf $MDS_DATA_PATH/fgvc-aircraft-2013b.tar.gz -C $MDS_DATA_PATH/
ls $MDS_DATA_PATH/fgvc-aircraft-2013b

# 3. conversion script
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=aircraft \
  --aircraft_data_root=$MDS_DATA_PATH/fgvc-aircraft-2013b \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

# 4. Expect the conversion to take 5 to 10 minutes.

# 5. Find the following outputs in $RECORDS/aircraft/:
#100 tfrecords files named [0-99].tfrecords
ls $RECORDS/aircraft/ | grep -c .tfrecords
#dataset_spec.json (see note 1)
cat $RECORDS/aircraft/dataset_spec.json





# -- cu_birds
ssh brando9@ampere4.stanford.edu
tmux new -s cu_birds
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

#1. download
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1 -O $MDS_DATA_PATH/CUB_200_2011.tgz
ls $MDS_DATA_PATH/

#2. extract
tar -xzf $MDS_DATA_PATH/CUB_200_2011.tgz -C $MDS_DATA_PATH/
ls $MDS_DATA_PATH/CUB_200_2011

#3. conversion
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=cu_birds \
  --cu_birds_data_root=$MDS_DATA_PATH/CUB_200_2011 \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#4. Expect the conversion to take around one minute.

#5. Find the following outputs in $RECORDS/cu_birds/:
#200 tfrecords files named [0-199].tfrecords
ls $RECORDS/cu_birds/ | grep -c .tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/cu_birds/dataset_spec.json



#-- dtd
ssh brando9@ampere4.stanford.edu
tmux new -s dtd
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

#1. download
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz -O $MDS_DATA_PATH/dtd-r1.0.1.tar.gz
ls $MDS_DATA_PATH/
#2. extract
tar xf $MDS_DATA_PATH/dtd-r1.0.1.tar.gz -C $MDS_DATA_PATH/
ls $MDS_DATA_PATH/dtd

#3. conversion
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=dtd \
  --dtd_data_root=$MDS_DATA_PATH/dtd \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#4. Expect the conversion to take a few seconds.

#5. Find the following outputs in $RECORDS/dtd/:
#47 tfrecords files named [0-46].tfrecords
ls $RECORDS/dtd/ | grep -c .tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/dtd/dataset_spec.json




# -- quickdraw
ssh brando9@ampere4.stanford.edu
tmux new -s quickdraw
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

#1. Download all 345 .npy files hosted on Google Cloud. You can use gsutil to download them to quickdraw/:
mkdir -p $MDS_DATA_PATH/quickdraw
gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy $MDS_DATA_PATH/quickdraw
# note, the download of these *.npy files takes some time, it's 36.8 GB, about 5 minutes see: ETA 00:04:23
# should show 345 *.npy files
ls $MDS_DATA_PATH/quickdraw
ls $MDS_DATA_PATH/quickdraw/ | grep -c .npy

#2. launch conversion script
#pip install numpy==1.23.1  # everything before this used 1.24.1 but quick draw needed 1.23.1, hopefully everything still works with 1.23.1, if not returning to 24 later, ref: https://stackoverflow.com/questions/74893742/how-to-solve-attributeerror-module-numpy-has-no-attribute-bool
pip list | grep numpy
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=quickdraw \
  --quickdraw_data_root=$MDS_DATA_PATH/quickdraw \
  --splits_root=$SPLITS \
  --records_root=$RECORDS
#pip install numpy==1.24.1  # return to 1.24.1
#pip list | grep numpy

# 3. Expect the conversion to take 3 to 4 hours. It also doesn't display well how much it's progressing

#4. Find the following outputs in $RECORDS/quickdraw/:
#345 tfrecords files named [0-344].tfrecords
ls $RECORDS/quickdraw/ | grep -c .tfrecords
#dataset_spec.json (see note 1)
cat $RECORDS/quickdraw/dataset_spec.json


# -- fungi
ssh brando9@ampere4.stanford.edu
tmux new -s fungi
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

#1. download
mkdir -p $MDS_DATA_PATH/fungi
wget https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz -O $MDS_DATA_PATH/fungi/fungi_train_val.tgz
wget https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz -O $MDS_DATA_PATH/fungi/train_val_annotations.tgz

ls $MDS_DATA_PATH/fungi

#2. Extract them into the same fungi/ directory. It should contain one images/ directory, as well as train.json and val.json.
tar -xzf $MDS_DATA_PATH/fungi/fungi_train_val.tgz -C $MDS_DATA_PATH/fungi
tar -xzf $MDS_DATA_PATH/fungi/train_val_annotations.tgz -C $MDS_DATA_PATH/fungi

# It should contain one images/ directory, as well as train.json and val.json.
ls $MDS_DATA_PATH/fungi

# 3. conversion
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=fungi \
  --fungi_data_root=$MDS_DATA_PATH/fungi \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

# 4. Expect the conversion to take 5 to 15 minutes.

#5. Find the following outputs in $RECORDS/fungi/:
#1394 tfrecords files named [0-1393].tfrecords
ls $RECORDS/fungi/ | grep -c .tfrecords
ls $RECORDS/fungi/
#dataset_spec.json (see note 1)
cat $RECORDS/fungi/dataset_spec.json



#-- vgg_flower
ssh brando9@ampere4.stanford.edu
tmux new -s vgg_flower
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

#1. download
mkdir -p $MDS_DATA_PATH/vgg_flower
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz -O $MDS_DATA_PATH/vgg_flower/102flowers.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat -O $MDS_DATA_PATH/vgg_flower/imagelabels.mat
# there should be two files
ls $MDS_DATA_PATH/vgg_flower/

#2. Extract 102flowers.tgz, it will create a jpg/ sub-directory
tar -xzf $MDS_DATA_PATH/vgg_flower/102flowers.tgz -C $MDS_DATA_PATH/vgg_flower
# it will create a jpg/ sub-directory
ls $MDS_DATA_PATH/vgg_flower
ls $MDS_DATA_PATH/vgg_flower/jpg

#3. conversion
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=vgg_flower \
  --vgg_flower_data_root=$MDS_DATA_PATH/vgg_flower \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#4. Expect the conversion to take about one minute.

#5. Find the following outputos in $RECORDS/vgg_flower/:
#102 tfrecords files named [0-101].tfrecords
ls $RECORDS/vgg_flower/ | grep -c .tfrecords
#dataset_spec.json (see note 1)
cat $RECORDS/vgg_flower/dataset_spec.json


#-- traffic_sign
ssh brando9@ampere4.stanford.edu
tmux new -s traffic_sign
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

#1. Download GTSRB_Final_Training_Images.zip If the link happens to be broken, browse the GTSRB dataset website for more information.
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip -O $MDS_DATA_PATH/GTSRB_Final_Training_Images.zip
ls $MDS_DATA_PATH/

#2. Extract it in $DATASRC, it will create a GTSRB/ sub-directory
unzip $MDS_DATA_PATH/GTSRB_Final_Training_Images.zip -d $MDS_DATA_PATH/
ls $MDS_DATA_PATH/

#3. conversion
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=traffic_sign \
  --traffic_sign_data_root=$MDS_DATA_PATH/GTSRB \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#4. Find the following outputs in $RECORDS/traffic_sign/:
#43 tfrecords files named [0-42].tfrecords
ls $RECORDS/traffic_sign/ | grep -c .tfrecords
#dataset_spec.json (see note 1)
cat $RECORDS/traffic_sign/dataset_spec.json




#-- mscoco
ssh brando9@ampere4.stanford.edu
tmux new -s mscoco
tmux new -s mscoco2
tmux new -s mscoco_zenodo
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

## 1. Download the 2017 train images and annotations from http://cocodataset.org/:
###You can use gsutil to download them to mscoco/:
##mkdir -p $MDS_DATA_PATH/mscoco/
##cd $MDS_DATA_PATH/mscoco/
##mkdir -p train2017
### seems to directly download all files, no zip file needed
##gsutil -m rsync gs://images.cocodataset.org/train2017 train2017
### todo should have 118287? number of .jpg files (note no unziping needed)
##ls $MDS_DATA_PATH/mscoco/train2017 | grep -c .jpg
### download & extract annotations_trainval2017.zip
##gsutil -m cp gs://images.cocodataset.org/annotations/annotations_trainval2017.zip
##unzip $MDS_DATA_PATH/mscoco/annotations_trainval2017.zip -d $MDS_DATA_PATH/mscoco
### todo says: 6?
##ls $MDS_DATA_PATH/mscoco/annotations | grep -c .json
#
## Download Otherwise, you can download train2017.zip and annotations_trainval2017.zip and extract them into mscoco/. eta ~36m.
#mkdir -p $MDS_DATA_PATH/mscoco
#wget http://images.cocodataset.org/zips/train2017.zip -O $MDS_DATA_PATH/mscoco/train2017.zip
#wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $MDS_DATA_PATH/mscoco/annotations_trainval2017.zip
## both zips should be there, note: downloading zip takes some time
#ls $MDS_DATA_PATH/mscoco/
## Extract them into mscoco/ takes (about ~5mins)
#unzip $MDS_DATA_PATH/mscoco/train2017.zip -d $MDS_DATA_PATH/mscoco
#ls $MDS_DATA_PATH/mscoco/train2017 | grep -c .jpg
## says: 118287 for a 2nd time
#unzip $MDS_DATA_PATH/mscoco/annotations_trainval2017.zip -d $MDS_DATA_PATH/mscoco
#ls $MDS_DATA_PATH/mscoco/annotations | grep -c .json
## says: 6 for a 2nd time
## move them since it says so in the google NL instructions ref: for moving large num files https://stackoverflow.com/a/75034830/1601580 thanks chatgpt!
#ls $MDS_DATA_PATH/mscoco/train2017 | grep -c .jpg
#find $MDS_DATA_PATH/mscoco/train2017 -type f -print0 | xargs -0 mv -t $MDS_DATA_PATH/mscoco
#ls $MDS_DATA_PATH/mscoco | grep -c .jpg
## says: 118287 for both
#ls $MDS_DATA_PATH/mscoco/annotations/ | grep -c .json
#mv $MDS_DATA_PATH/mscoco/annotations/* $MDS_DATA_PATH/mscoco/
#ls $MDS_DATA_PATH/mscoco/ | grep -c .json
## says: 6 for both
# - download mscoco using Zenodo
cd $RECORDS/

#download the records of mscoco and uncompress. took ~1hr (5.1GB) since my wifi is slow but maybe faster for others?
wget https://zenodo.org/record/7517539/files/mscoco.tar.gz?download=1 -P $RECORDS -O mscoco.tar.gz
# should (5.1GB)
ls -lh $RECORDS/mscoco.tar.gz

# unziping is quick
tar -xf $RECORDS/mscoco.tar.gz -C $RECORDS/
mv $RECORDS/home/pzy2/data/mds/records/mscoco $RECORDS/mscoco
# should be (X GHB) todo
ls -lh $RECORDS/mscoco

#get splits file (should take a few seconds)
wget https://zenodo.org/record/7517539/files/mscoco_splits.json?download=1 -o $SPLITS/mscoco_splits.json
cat $SPLITS/mscoco_splits.json

# Find the following outputs in $RECORDS/mscoco/:
#80 tfrecords files named [0-79].tfrecords
ls $RECORDS/mscoco/ | grep -c .tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/mscoco/dataset_spec.json
#this should be X GB if everything goes to plan. todo
ls -h -sh $RECORDS/mscoco/

# -- TODO other datasets (fungi quickdraw ilsvrc mscoco omniglot), haven't verified they worked yet in my setup
# try running the old commands from https://github.com/brando90/diversity-for-predictive-success-of-meta-learning/blob/main/download_meta_dataset_mds.sh
# and let me know which one doesnt work via gitissue


# final step - run make_index_files.sh
cd $HOME/pytorch-meta-dataset/
chmod +x make_index_files.sh
./make_index_files.sh

# 2. Launch the conversion script:
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=mscoco \
  --mscoco_data_root=$MDS_DATA_PATH/mscoco \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

# 3. Expect the conversion to take about 4 hours.

# 4. Find the following outputs in $RECORDS/mscoco/:
#80 tfrecords files named [0-79].tfrecords
ls $RECORDS/mscoco/ | grep -c .tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/mscoco/dataset_spec.json

# -- final step - run make_index_files.sh
cd $HOME/pytorch-meta-dataset/
chmod +x make_index_files.sh
./make_index_files.sh



# --- tfds: last attempt if above didn't work: https://github.com/google-research/meta-dataset/blob/main/meta_dataset/data/tfds/README.md
ssh brando9@ampere4.stanford.edu

krbtmux
reauth
source $AFS/.bashrc.lfs
conda activate mds_env_gpu

tmux new -s tfds
tmux attach -t tfds

# - start again (CAREFUL!)
# rm -rf $HOME/tensorflow_datasets/meta_dataset

# - set up the file descriptor limit or the remaining tfds cmds won't work
# check soft limit, output should be 1024
ulimit -Sn
# check hard limit, output should be 1024
ulimit -Hn
# increase the limit
ulimit -n 120000
# check soft limit, output should be 120000
ulimit -Sn
# check hard limit, output should be 120000
ulimit -Hn

# - The only manual intervention required is to download the ILSVRC 2012 training data (ILSVRC2012_img_train.tar) into TFDS's manual download directory (e.g. ~/tensorflow_datasets/downloads/manual/).
# (ILSVRC2012_img_train.tar) into TFDS's manual download directory (e.g. ~/tensorflow_datasets/downloads/manual/).
mkdir -p $HOME/tensorflow_datasets/downloads/manual/
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -O  $HOME/tensorflow_datasets/downloads/manual/ILSVRC2012_img_train.tar
# check size of .tar file, should 138G
ls -lh $HOME/tensorflow_datasets/downloads/manual/ILSVRC2012_img_train.tar

# - First, make sure that meta_dataset and its dependencies are installed. This can be done with ... one of the approaches at the top of this file. Not copy pasting to avoid maintaining two different set of codes
# pip install & reqs.txt...

# - Generating the tfrecord files associated with all data sources and storing them in ~/tensorflow_datasets/meta_dataset is done
# check size of .tar file, should 138G
ls -lh $HOME/tensorflow_datasets/downloads/manual/ILSVRC2012_img_train.tar
# check that the imganet tar file is where it should be i.e. the <MANUAL_DIR> is the directory where the ILSVRC2012_img_train.tar file was downloaded. Check that it's there and it's 13GB (according to original instructions)
ls -lh $HOME/tensorflow_datasets/downloads/manual
# cd into the tfds dir & run the required command
cd $HOME/diversity-for-predictive-success-of-meta-learning/meta-dataset/meta_dataset/data/tfds
tfds build md_tfds --manual_dir=$HOME/tensorflow_datasets/downloads/manual
# todo: this step takes... hours (started 8pm jan 10th...ended...)

# - Check that tfds created the data set correctly by checking the tfrecord files associated with all data sources are at $HOME/tensorflow_datasets/meta_dataset
# check the folders are the data sets you expect & the sizes
ls -lh $HOME/tensorflow_datasets/meta_dataset
# todo check all tfrecords count for each data set
#ls ls $HOME/tensorflow_datasets/meta_dataset/<eachdataset> | grep -c .tfrecords
# a. imagnet todo
ls $RECORDS/ilsvrc_2012/ | grep -c .tfrecords
ls $RECORDS/ilsvrc_2012/dataset_spec.json
ls $RECORDS/ilsvrc_2012/num_leaf_images.json
# b. omniglot todo
ls $RECORDS/omniglot/ | grep -c .tfrecords
ls $RECORDS/omniglot/dataset_spec.json
# c. aircraft todo
ls $RECORDS/aircraft/ | grep -c .tfrecords
ls $RECORDS/aircraft/dataset_spec.json
# d. cu_birds todo
ls $RECORDS/cu_birds/ | grep -c .tfrecords
ls $RECORDS/cu_birds/dataset_spec.json
# e. dtd todo
ls $RECORDS/dtd/ | grep -c .tfrecords
ls $RECORDS/dtd/dataset_spec.json
# f. quickdraw todo
ls $RECORDS/quickdraw/ | grep -c .tfrecords
ls $RECORDS/quickdraw/dataset_spec.json
# g. fungi todo
ls $RECORDS/fungi/ | grep -c .tfrecords
ls $RECORDS/fungi/dataset_spec.json
# h. vgg_flowers todo
ls $RECORDS/vgg_flowers/ | grep -c .tfrecords
ls $RECORDS/vgg_flowers/dataset_spec.json
# i. traffic_sign todo
ls $RECORDS/traffic_sign/ | grep -c .tfrecords
ls $RECORDS/traffic_sign/dataset_spec.json
# j. mscoco todo
ls $RECORDS/mscoco/ | grep -c .tfrecords
ls $RECORDS/mscoco/dataset_spec.json

# todo: decide what to do with this...move it to where the the original instructions without tfds say to move them to? ask patrick how it relates to: $MDS_DATA_PATH, $RECORDS and $SPLITS.
#mv $HOME/tensorflow_datasets/meta_dataset/* $HOME/data/mds_tfds

