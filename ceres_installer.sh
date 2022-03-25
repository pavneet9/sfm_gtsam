sudo apt update -y && sudo apt upgrade -y

sudo apt-get install libgoogle-glog-dev libgflags-dev -y
sudo apt-get install libatlas-base-dev -y
sudo apt-get install libeigen3-dev -y
sudo apt-get install libsuitesparse-dev -y

# Download
cd /tmp
CERES_VERSION="ceres-solver-2.0.0"
CERES_ARCHIVE="$CERES_VERSION.tar.gz"
wget http://ceres-solver.org/$CERES_ARCHIVE
tar xfv $CERES_ARCHIVE

# Install
cd $CERES_VERSION
mkdir build
cd build
NUM_CPU_CORES=$(grep -c ^processor /proc/cpuinfo)
cmake ..
cmake --build . -j $NUM_CPU_CORES

sudo apt install checkinstall libssl-dev -y
sudo checkinstall --pkgname ceres-solver
