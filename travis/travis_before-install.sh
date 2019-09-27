#!/bin/bash
#
# This script is run in the "before_install" section of the travis file.
# It is also used when building dependencies inside a travis docker container
# for debugging.

# Set the C and C++ compilers
echo "  CC = ${CC} $(${CC} -dumpversion)"
echo "  CXX = ${CXX} $(${CXX} -dumpversion)"

# Set python install prefix and site version
export PYPREFIX=$(dirname $(dirname $(which python3)))
echo "Python install prefix = ${PYPREFIX}"
export PYSITE=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
echo "Python site version = ${PYSITE}"

# Find the path to the python headers and libs, since cmake can't
export PYROOT=$(python-config --prefix)
export PYINC=$(python-config --includes | awk '{print $1}' | sed -e "s#-I##")
export PYLIB=${PYROOT}/lib/$(python-config --libs | awk '{print $1}' | sed -e "s#-l#lib#").so

# Our current working directory
export WORKDIR=$(pwd)

# # Install python dependencies.
# pip3 install numpy scipy matplotlib cmake cython astropy ephem healpy numba toml quaternionarray \
# && pip3 install https://github.com/healpy/pysm/archive/master.tar.gz \
# && pip3 install https://github.com/hpc4cmb/toast/archive/master.tar.gz
# if [ $? -ne 0 ]; then
#     echo "pip install failed" >&2
#     exit 1
# fi
#
# # Fetch patch that disables non-portable compile flags in spt3g.
# curl -SL https://raw.githubusercontent.com/hpc4cmb/cmbenv/master/pkgs/patch_spt3g -o patch_spt3g
# if [ $? -ne 0 ]; then
#     echo "curl failed" >&2
#     exit 1
# fi
#
# # Install spt3g to /usr, and python bindings into python prefix
# export BOOST_ROOT=/usr
# git clone --branch master --single-branch --depth 1 https://github.com/CMB-S4/spt3g_software.git \
# && cd spt3g_software \
# && patch -p1 < ../patch_spt3g \
# && mkdir build \
# && cd build \
# && LDFLAGS="-Wl,-z,muldefs" \
#     cmake \
#     -DCMAKE_C_COMPILER="${CC}" \
#     -DCMAKE_CXX_COMPILER="${CXX}" \
#     -DCMAKE_C_FLAGS="-g -O2 -fPIC -pthread" \
#     -DCMAKE_CXX_FLAGS="-g -O2 -fPIC -pthread -std=c++11" \
#     -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
#     -DBOOST_ROOT="/usr" \
#     -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
#     -DPYTHON_INCLUDE_DIR="${PYINC}" \
#     -DPYTHON_LIBRARY="${PYLIB}" \
#     .. \
# && make -j 2 \
# && sudo ln -s ${WORKDIR}/spt3g_software/build/bin/* /usr/bin/ \
# && ln -s ${WORKDIR}/spt3g_software/build/spt3g ${PYPREFIX}/lib/python${PYSITE}/site-packages/
# if [ $? -ne 0 ]; then
#     echo "spt3g install failed" >&2
#     exit 1
# fi
#
# cd ${WORKDIR}

# Install so3g to /usr.  We have to strip out the "-Werror" option because
# the code does not compile without warnings on all gcc versions.
export BOOST_ROOT=/usr

export SPT3G_SOFTWARE_PATH="${WORKDIR}/spt3g_software"
export SPT3G_SOFTWARE_BUILD_PATH="${WORKDIR}/spt3g_software/build"
git clone --branch master --single-branch --depth 1 https://github.com/simonsobs/so3g.git \
&& cd so3g \
&& sed -i -e "s#-Werror##g" CMakeLists.txt \
&& mkdir build \
&& cd build \
&& LDFLAGS="-Wl,-z,muldefs" \
    cmake \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="-g -O2 -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-g -O2 -fPIC -pthread -std=c++11" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DPYTHON_INSTALL_DEST=${PYPREFIX}/lib/python${PYSITE}/site-packages \
    .. \
&& make -j 2 \
&& sudo make install
if [ $? -ne 0 ]; then
    echo "so3g install failed" >&2
    exit 1
fi

cd ${WORKDIR}
