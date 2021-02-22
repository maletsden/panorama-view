# panorama-view
Fast Panorama Stitching for Image Sequence

### Install required dependencies
```bash
sudo apt-get install libjpeg-dev
sudo apt install libeigen3-dev
```

### Install required Google Test
```bash
sudo apt-get install libgtest-dev
cd /usr/src/gtest
sudo cmake CMakeLists.txt
make
sudo cp ./lib/libgtest*.a /usr/local/lib
```

### Run Panorama Stitching
```bash
cmake -B build . && \
make --directory=build && \
./build/panorama_view
```

### Run test for Panorama Stitching
```bash
cmake -B build . && \
make --directory=build && \
./build/panorama_view_tester
```