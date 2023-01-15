# I want to send a compressed tar.gz file located locally at path /Users/patrickyu/Documents/ilsvrc.tar.gz and want to send it to the server brando9@ampere4.stanford.edu and in the server the path is /lfs/ampere4/0/brando9/data/mds, how do I do this in bash?
#scp /Users/patrickyu/Documents/ilsvrc.tar.gz brando9@ampere4.stanford.edu:/lfs/ampere4/0/brando9/data/mds
scp /Users/patrickyu/Documents/ilsvrc.tar.gz brando9@ampere4.stanford.edu:/lfs/ampere4/0/brando9/data/mds/records/
echo $RECORDS

# check the size of the file using ls
ls -lh /lfs/ampere4/0/brando9/data/mds/records/ilsvrc.tar.gz

#check that the md5hash matches my (working) tar.gz file
md5sum $RECORDS/ilsvrc.tar.gz #should be 56c576d10896bfa8d35200aebfea1704

#should have ilsvrc.tar.gz in $RECORDS/
#now extract it
tar -xf $RECORDS/ilsvrc.tar.gz -C $RECORDS/

# Need to un-nest folders since I extracted my mscoco at the top-most directory instead of in $RECORDS/
mv $RECORDS/shared/rsaas/pzy2/records/ilsvrc_2012  $RECORDS/ilsvrc_2012

# remove dir $RECORDS/shared/rsaas/pzy2/records/
rmdir $RECORDS/shared/rsaas/pzy2/records/