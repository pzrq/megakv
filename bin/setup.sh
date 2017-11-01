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

# Note: This should be on a second distinct AWS machine
# TODO: Verify Step 7 - Benchmark works
cd ${MEGAKV_HOME}/benchmark
RTE_SDK="$HOME/build/dpdk" make  # RTE_SDK value assumed to be true from Step 1

echo "\e[32m DPDK, libgpuhash, MegaKV and benchmark have built successfully."
echo "\e[32m The following commands should work:"
echo ""
echo "\e[32m   # Run MegaKV GPU test - should see 100% GPU util on nvidia-smi"
echo "\e[32m   ./libgpuhash/test/run \e[0m"
echo ""
echo "\e[32m   # Run MegaKV itself"
echo "\e[32m   # should see 4 x 100% CPU on htop, 1-5% GPU util on nvidia-smi"
echo "\e[32m   ./src/build/app/megakv"
echo ""
echo "\e[32m   # Make and effect a performance tuning change"
echo "\e[32m   # In src/macros.h, "
echo "\e[32m   # change the values below from 7 and 12 to 1 and 1"
echo "\e[32m   #       define NUM_QUEUE_PER_PORT	7"
echo "\e[32m   #       define MAX_WORKER_NUM		12"
echo "\e[32m   # Then recompile MegaKV with:"
echo "\e[32m   cd src"
echo "\e[32m   RTE_SDK="$HOME/build/dpdk" make"
echo "\e[32m   cd ../"
echo "\e[32m   # Re-running MegaKV itself"
echo "\e[32m   # should see 3 x 100% CPU on htop, 30-10% GPU util on nvidia-smi"
echo "\e[32m   ./src/build/app/megakv"
echo ""
echo "\e[32m   # Run MegaKV's benchmark, "
echo "\e[32m   # preferably on a separate EC2 instance pointed to MegaKV"
echo "\e[32m   sudo ./benchmark/build/app/megakv \e[0m"
