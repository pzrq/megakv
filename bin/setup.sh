#!/usr/bin/env bash

wget http://kay21s.github.io/megakv/megakv-0.1-alpha.tar.gz
tar -xvf megakv-0.1-alpha.tar.gz

# Intel DPDK Setup - Following this guide
# https://www.slideshare.net/hisaki/intel-dpdk-step-by-step-instructions

# 1. Combine several approaches to enable HugePages
# https://access.redhat.com/documentation/en-US/Red_Hat_Enterprise_Linux/5/html/Tuning_and_Optimizing_Red_Hat_Enterprise_Linux_for_Oracle_9i_and_10g_Databases/sect-Oracle_9i_and_10g_Tuning_Guide-Large_Memory_Optimization_Big_Pages_and_Huge_Pages-Configuring_Huge_Pages_in_Red_Hat_Enterprise_Linux_4_or_5.html
# https://askubuntu.com/questions/776929/how-to-edit-my-etc-sysctl-conf-file
sudo mkdir /hugepages
sudo sysctl -w vm.nr_hugepages=256
# TODO: This is not idempotent yet, needs a guard
echo "" | sudo tee -a /etc/sysctl.conf
echo "# Enable hugepages for Intel DPDK" | sudo tee -a /etc/sysctl.conf
echo "vm.nr_hugepages=256" | sudo tee -a /etc/sysctl.conf
cat /etc/fstab
# Should now report 256
grep HugePages_Total /proc/meminfo

# Note 1.8, not 1.7.1 as recommended in README
#wget http://fast.dpdk.org/rel/dpdk-1.8.0.tar.xz
#tar -xvf dpdk-1.8.0.tar.xz
#cd dpdk-1.8.0
#make config T=x86_64-native-linuxapp-gcc && make

# https://github.com/att/vfd/wiki/Building-the-igb_uio-driver
sudo yum install gcc git gmake kernel-devel -y
mkdir $HOME/build
cd $HOME/build
git clone http://dpdk.org/git/dpdk
cd dpdk
# TODO: This is not v1.7.1 the MegaKV author recommended - will it work?
git checkout v16.11
make config T=x86_64-native-linuxapp-gcc

# On the AWS P2 instance
# Choose option 13 x86_64-native-linuxapp-gcc
# (lots of messages/wait for 2-5 minutes)
# Then choose option 16 (Insert IGB UIO module) and 33 (Exit Script)
tools/dpdk-setup.sh
# TODO: Installation cannot run with T defined and DESTDIR undefined
# TODO: ... ^ Check if the above is an issue or not?

# TODO: Takes 2-5 minutes? Might be a duplicate of the 'tools/dpdk-setup.sh' above...
RTE_SDK=`pwd` make

pwd
# /home/ec2-user/build/dpdk

#sudo make install T=x86_64-native-linuxapp-gcc
