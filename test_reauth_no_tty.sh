#!/usr/bin/expect -f

##!/bin/bash
##!/usr/bin/expect -f

# expect
# send

# https://unix.stackexchange.com/questions/724902/how-does-one-send-new-commands-to-run-to-an-already-running-nohup-process-e-g-r

# - CAREFUL, if a job is already running it could do damage to it, rm reauth process, qian doesn't do it so skip it
# top -u brando9
# pkill -9 reauth -u brando9;
# pkill -9 reauth -u brando9; pkill -9 python -u brando9; pkill -9 wandb-service*
# pkill -9 tmux -u brando9

# stty -echo or stty echo
source /etc/bashrc
source /etc/profile
source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile
source ~/.bashrc.user
stty echo

# - test reauth
echo
echo $SU_PASSWORD | /afs/cs/software/bin/reauth
#tty echo $SU_PASSWORD | /afs/cs/software/bin/reauth

sh -ic 'echo $SU_PASSWORD | /afs/cs/software/bin/reauth'

sleep 1
pkill -9 reauth -u brando9
