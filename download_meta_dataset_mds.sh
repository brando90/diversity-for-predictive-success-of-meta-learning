# based on: https://github.com/google-research/meta-dataset/blob/main/doc/dataset_conversion.md
# make index is needed, it's at the last step at the end of this doc (after installing all the tfrecords from MDS)

# -- prereqs: install gsutil
# if that doesnt work here's googles OS-specific instructions to install gsutil: https://download.huihoo.com/google/gdgdevkit/DVD1/developers.google.com/storage/docs/gsutil_install.html
# see "Installing from the Python package index (PyPi)"
pip install gsutil

# -- prereqs: download pytorch-mds to $HOME
cd $HOME
git clone https://github.com/brando90/pytorch-meta-dataset # done already?
#check that the pytorch-mds directory is there
ls pytorch-meta-dataset/

# -- prereqs: download original mds to $HOME
git clone https://github.com/google-research/meta-dataset
#check that the original mds directory is there
ls meta-dataset/

# -- prereqs: install original mds python requirements
pip install -r meta-dataset/requirements.txt

# -- prereqs: install pytorch mds python requirements
pip install -r pytorch-meta-dataset/requirements.txt

#create records and splits folders for mds
mkdir -p $HOME/data/mds #changed from $HOME/mds to $HOME/data/mds?
export MDS_DATA_PATH=$HOME/data/mds

mkdir $MDS_DATA_PATH/records # If you haven't already?
mkdir $MDS_DATA_PATH/splits # If you haven't already?
export RECORDS=$MDS_DATA_PATH/records
export SPLITS=$MDS_DATA_PATH/splits

# in order to run python scripts we need to cd into the original mds dir
cd $HOME/meta-dataset/

# -- ilsvrc_2012: https://github.com/google-research/meta-dataset/blob/main/doc/dataset_conversion.md#ilsvrc_2012
# - 1. Download ilsvrc2012_img_train.tar, from the ILSVRC2012 website
# todo: https://gist.github.com/bonlime/4e0d236cf98cd5b15d977dfa03a63643
# todo: https://github.com/google-research/meta-dataset/blob/main/doc/dataset_conversion.md#ilsvrc_2012
# wget TODO -O $MDS_DATA_PATH/ilsvrc_2012
# for imagenet url: https://image-net.org/download-images.php
wget https://image-net.org/data/winter21_whole.tar.gz -O ~/data/winter21_whole.tar.gz

# - 2. Extract it into ILSVRC2012_img_train/, which should contain 1000 files, named n????????.tar (expected time: ~30 minutes)
# tar https://superuser.com/questions/348205/how-do-i-unzip-a-tar-gz-archive-to-a-specific-destination
mkdir -p ~/data/winter21_whole
tar xf ~/data/winter21_whole.tar.gz -C ~/data/winter21_whole
# move the train part: mv src dest
mv ~/data/winter21_whole/ILSVRC2012_img_train $MDS_DATA_PATH/ILSVRC2012_img_train

# - 3. Extract each of ILSVRC2012_img_train/n????????.tar in its own directory (expected time: ~30 minutes), for instance:
for FILE in $MDS_DATA_PATH/ILSVRC2012_img_train/*.tar;
do
  #echo $FILE
  mkdir ${FILE/.tar/};
  cd ${FILE/.tar/};
  tar xvf ../$FILE;
  cd ..;
done

# - 4. Download the following two files into ILSVRC2012_img_train/
wget http://www.image-net.org/data/wordnet.is_a.txt -O $MDS_DATA_PATH/ILSVRC2012_img_train/wordnet.is_a.txt
wget http://www.image-net.org/data/words.txt -O $MDS_DATA_PATH/ILSVRC2012_img_train/words.txt

# - 5. Launch the conversion script (Use --dataset=ilsvrc_2012_v2 for the training only MetaDataset-v2 version):
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=ilsvrc_2012 \
  --ilsvrc_2012_data_root=$MDS_DATA_PATH/ILSVRC2012_img_train \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

# -6. Expect the conversion to take 4 to 12 hours, depending on the filesystem's latency and bandwidth.
#nop

# - 7.Find the following outputs in $RECORDS/ilsvrc_2012/:
      #
      #1000 tfrecords files named [0-999].tfrecords
      #dataset_spec.json (see note 1)
      #num_leaf_images.json
ls $RECORDS/ilsvrc_2012/




# -- omniglot: https://github.com/google-research/meta-dataset/blob/main/doc/dataset_conversion.md#omniglot
# 1. download omniglot
mkdir $MDS_DATA_PATH/omniglot/
wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip -O $MDS_DATA_PATH/omniglot/images_background.zip
wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip -O $MDS_DATA_PATH/omniglot/images_evaluation.zip
unzip $MDS_DATA_PATH/omniglot/images_background.zip -d $MDS_DATA_PATH/omniglot
unzip $MDS_DATA_PATH/omniglot/images_evaluation.zip -d $MDS_DATA_PATH/omniglot
# what is
#   --splits_root=$SPLITS \
#   --records_root=$RECORD
# SPLITS is what does the hierarchical splitting of classes when sampling tasks (you should have a few json files),
# RECORDS is where the tfrecords are recorded. plus a json file dataspec to map classes to json files.

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=omniglot \
  --omniglot_data_root=$MDS_DATA_PATH/omniglot \
  --splits_root=$SPLITS \
  --records_root=$RECORDS



# -- aircraft: https://github.com/google-research/meta-dataset/blob/main/doc/dataset_conversion.md#aircraft
# 1. download and extract
wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz -O $MDS_DATA_PATH/fgvc-aircraft-2013b.tar.gz
tar xf $MDS_DATA_PATH/fgvc-aircraft-2013b.tar.gz -C $MDS_DATA_PATH/

# 2. conversion script
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=aircraft \
  --aircraft_data_root=$MDS_DATA_PATH/fgvc-aircraft-2013b \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#3. Find the following outputs in $RECORDS/aircraft/:

#100 tfrecords files named [0-99].tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/aircraft/


# -- cu_birds
#1. download+extract
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1 -O $MDS_DATA_PATH/CUB_200_2011.tgz
tar -xzf $MDS_DATA_PATH/CUB_200_2011.tgz -C $MDS_DATA_PATH/

# 2. conversion
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=cu_birds \
  --cu_birds_data_root=$MDS_DATA_PATH/CUB_200_2011 \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#3. Find the following outputs in $RECORDS/cu_birds/:

#200 tfrecords files named [0-199].tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/cu_birds/



#-- dtd
#1. download+extract
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz -O $MDS_DATA_PATH/dtd-r1.0.1.tar.gz
tar xf $MDS_DATA_PATH/dtd-r1.0.1.tar.gz -C $MDS_DATA_PATH/

# 2. conversion
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=dtd \
  --cu_birds_data_root=$MDS_DATA_PATH/dtd \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#Find the following outputs in $RECORDS/dtd/:

#47 tfrecords files named [0-46].tfrecords
#dataset_spec.json (see note 1)

ls $RECORDS/dtd/


# -- quickdraw

# --1. run gsutil to get the files
gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy $MDS_DATA_PATH/quickdraw

# --2. launch conversion script
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=quickdraw \
  --quickdraw_data_root=$MDS_DATA_PATH/quickdraw \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#--3. Find the following outputs in $RECORDS/quickdraw/:
#345 tfrecords files named [0-344].tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/quickdraw/


# -- fungi
#1. download+extract
mkdir $MDS_DATA_PATH/fungi
wget https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz -O $MDS_DATA_PATH/fungi/fungi_train_val.tgz
wget https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz -O $MDS_DATA_PATH/fungi/train_val_annotations.tgz
tar -xzf $MDS_DATA_PATH/fungi/fungi_train_val.tgz -C $MDS_DATA_PATH/fungi
tar -xzf $MDS_DATA_PATH/fungi/train_val_annotations.tgz -C $MDS_DATA_PATH/fungi

# 2. conversion
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=fungi \
  --fungi_data_root=$MDS_DATA_PATH/fungi \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#Find the following outputs in $RECORDS/fungi/:

#1394 tfrecords files named [0-1393].tfrecords
#dataset_spec.json (see note 1)

ls $RECORDS/fungi/

#-- vgg_flower
#1. download+extract
mkdir $MDS_DATA_PATH/vgg_flower
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz -O $MDS_DATA_PATH/vgg_flower/102flowers.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat -O $MDS_DATA_PATH/vgg_flower/imagelabels.mat
tar -xzf $MDS_DATA_PATH/vgg_flower/102flowers.tgz -C $MDS_DATA_PATH/vgg_flower


#2. conversion
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=vgg_flower \
  --vgg_flower_data_root=$MDS_DATA_PATH/vgg_flower \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#Find the following outputs in $RECORDS/vgg_flower/:

#102 tfrecords files named [0-101].tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/vgg_flower/

#-- traffic_sign
#1, dwld+extract
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip -O $MDS_DATA_PATH/GTSRB_Final_Training_Images.zip
unzip $MDS_DATA_PATH/GTSRB_Final_Training_Images.zip -d $MDS_DATA_PATH/

#2. conversion
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=traffic_sign \
  --traffic_sign_data_root=$MDS_DATA_PATH/GTSRB \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#3. Find the following outputs in $RECORDS/traffic_sign/:

#43 tfrecords files named [0-42].tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/traffic_sign/

#--mscoco
#1, dwld+extract
mkdir $MDS_DATA_PATH/mscoco
wget http://images.cocodataset.org/zips/train2017.zip -O $MDS_DATA_PATH/mscoco/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $MDS_DATA_PATH/mscoco/annotations_trainval2017.zip
unzip $MDS_DATA_PATH/mscoco/annotations_trainval2017.zip -d $MDS_DATA_PATH/mscoco
unzip $MDS_DATA_PATH/mscoco/train2017.zip -d $MDS_DATA_PATH/mscoco

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=mscoco \
  --mscoco_data_root=$MDS_DATA_PATH/mscoco \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

#Find the following outputs in $RECORDS/mscoco/:

#80 tfrecords files named [0-79].tfrecords
#dataset_spec.json (see note 1)
ls $RECORDS/mscoco/

# final step - run make_index_files.sh
cd $HOME/pytorch-meta-dataset/
chmod +x make_index_files.sh
./make_index_files.sh