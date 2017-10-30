#!/usr/bin/env bash

# Run on an AWS p2.xlarge instance with the ami-5af95d20,
# "Deep Learning AMI CUDA 9 Ubuntu Version".
#
# On instance login, it reports:
# > Welcome to Ubuntu 16.04.3 LTS (GNU/Linux 4.4.0-1022-aws x86_64)
#
# ubuntu@ip-zzz-zzz-zzz-zzz:~/src$ ls
# anaconda2  caffe2                      tensorflow_anaconda2  tensorflow_python3
# anaconda3  caffe2_anaconda2  mxnet     tensorflow_anaconda3
# bin        logs              OpenBLAS  tensorflow_python2
#
# See https://aws.amazon.com/marketplace/pp/B076TGJHY1
#
# This script assumes the following manual setup has been completed:
#
#     sudo apt-get install git
#     cd $HOME
#     git clone https://github.com/pzrq/megakv.git megakv

# Change if you cloned the repo somewhere else - YMMV (PRs welcome)
MEGAKV_HOME="$HOME/megakv"

# TODO: Step 0 - Possibly fix some problems with the AMI (do we need this?)
apt-get -f install
apt-get install libdpdk0 libxenstore3.0 dpdk

# Step 1 - Set up Intel Data Plane Development Kit(DPDK) for network performance testing
${MEGAKV_HOME}/bin/dpdk_setup.sh

# TODO: Steps 2 & 3 - Verify if this works from scratch, see README
# TODO: ... e.g. CUDA installation path is?
cd ${MEGAKV_HOME}/libgpuhash
make

# TODO: Step 4 - Same as previous steps - CUDA installation path?
cd ${MEGAKV_HOME}/src
RTE_SDK="$HOME/build/dpdk" make  # RTE_SDK value assumed to be true from Step 1
