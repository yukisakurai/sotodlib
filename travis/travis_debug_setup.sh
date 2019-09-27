#!/bin/bash
#
# This script is designed to be run *INSIDE* a docker container that has
# been launched like:
#
#    %> ./travis_debug_start.sh <instance ID>
#
# Once you are inside the docker container, you can then run this script with
#
#    %> ./scripts/travis_debug_setup.sh <python version> <gcc suffix>
#
# Where "gcc suffix" is something like "7", "5", or "4.9" and python versions
# is something like "3.5" or "3.6".  This will install dependencies to match
# what is done in the travis file.
#

usage () {
    echo "$0 <python version> <toolchain suffix>"
    exit 1
}

pyversion="$1"
gccversion="$2"

# Runtime packages.  These are packages that will be installed into system
# locations when travis is actually running.

# Install APT packages
sudo -E apt-add-repository -y "ppa:ubuntu-toolchain-r/test"
sudo -E apt-get -yq update
sudo -E apt-get -yq --no-install-suggests --no-install-recommends install \
build-essential \
git \
curl \
pkg-config \
locales \
libgl1-mesa-glx \
xvfb \
libboost-all-dev \
libopenblas-dev \
liblapack-dev \
libfftw3-dev \
libflac-dev \
libhdf5-dev \
gcc-7 \
g++-7 \
gcc-${gccversion} \
g++-${gccversion}

# Export serial compiler variables
export CC=$(which gcc-${gccversion})
export CXX=$(which g++-${gccversion})

# Install travis python
if [ ! -e "${HOME}/virtualenv/python3.6/bin/activate" ]; then
    wget https://s3.amazonaws.com/travis-python-archives/binaries/ubuntu/14.04/x86_64/python-${pyversion}.tar.bz2
    sudo tar xjf python-${pyversion}.tar.bz2 --directory /
fi
source ${HOME}/virtualenv/python${pyversion}/bin/activate

# Install the remaining dependencies
./scripts/travis_before-install.sh
