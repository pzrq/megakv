#!/usr/bin/env bash

# Intel Data Plane Development Kit(DPDK) - Following this guide
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

# https://github.com/att/vfd/wiki/Building-the-igb_uio-driver
sudo apt-get update
sudo apt-get install gcc git gmake linux-headers-generic -y
# Red Hat version...
#sudo yum install gcc git gmake kernel-devel -y

mkdir $HOME/build
cd $HOME/build
git clone http://dpdk.org/git/dpdk
cd dpdk
# TODO: This is not v1.7.1 the MegaKV author recommended - will it work?
# Note that v1.7.1 gives a bunch of errors like:
# /home/ubuntu/build/dpdk/mk/toolchain/gcc/rte.toolchain-compat.mk:46: You are not using GCC 4.x. This is neither supported, nor tested.
# /usr/src/linux-headers-4.4.0-1022-aws/Makefile:1420: recipe for target '_module_/home/ubuntu/build/dpdk/build/build/lib/librte_eal/linuxapp/igb_uio' failed
#
# A list of DPDK versions is available at:
# http://dpdk.org/doc/guides/rel_notes/
#
# See also the "Build Directory Concept"
# http://dpdk.org/doc/guides/prog_guide/dev_kit_build_system.html
# Note: Using v16.11 over v17.08 as the former felt like it compiled more quickly
#
# v16.11 timing:
# Build complete [x86_64-native-linuxapp-gcc]
# 134.28user 15.80system 2:54.72elapsed 85%CPU (0avgtext+0avgdata 176284maxresident)k
# 0inputs+234104outputs (0major+10168375minor)pagefaults 0swaps
#
# v17.08 timing?
# Build complete [x86_64-native-linuxapp-gcc]
# 144.03user 19.50system 3:16.87elapsed 83%CPU (0avgtext+0avgdata 167300maxresident)k
# 0inputs+247560outputs (0major+12937461minor)pagefaults 0swaps
git checkout v16.11

# Define RTE_TARGET to work with MegaKV's defaults
RTE_TARGET="x86_64-native-linuxapp-gcc"
make config T=x86_64-native-linuxapp-gcc O=${RTE_TARGET}

# Current working directory should be $HOME/build/dpdk, so downstream tools
# can use RTE_SDK="$HOME/build/dpdk"
echo `pwd`

# This part takes about 2-5 minutes to compile stuff
RTE_SDK=`pwd` make O=${RTE_TARGET}

echo "\e[32m DPDK installation should have completed successfully."
echo "\e[32m You should now be able to use `RTE_SDK="$HOME/build/dpdk"` \e[0m"
