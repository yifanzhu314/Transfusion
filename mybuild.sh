cd ./Core
rm -rf build/*
# mkdir build
cd build
cmake ../src
make -j28
cd ../../GPUTest
rm -rf build/*
# mkdir build
cd build
cmake ../src
make -j28
cd ../../GUI
rm -rf build/*
# mkdir build
cd build
cmake  ../src -DCMAKE_PREFIX_PATH=/home/zyf-lab/program/libtorch/ -D_GLIBCXX_USE_CXX11_ABI=0
# ccmake ../src
make -j28
# cd /home/mathloverpi/code/ElasticFusion/GUI/build
#./ElasticFusion -l ../4.klg
