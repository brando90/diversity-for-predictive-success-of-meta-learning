# - Rok's suggestion, sending data from ampere4 to ampere3
cd ~/data/mds; rsync -avW records brando9@ampere3:/lfs/local/0/brando9

# - send data from ampere4 to ampere3 using rsync and creating dir if it doesn't exist
cd ~/data/mds; rsync -avW records brando9@ampere1:/lfs/local/0/brando9/data/mds
cd ~/data/mds; rsync -avW records brando9@ampere2:/lfs/local/0/brando9/data/mds
cd ~/data/mds; rsync -avW records brando9@ampere3:/lfs/local/0/brando9/data/mds


# - send l2l to ampere3
# todo idk if needed

