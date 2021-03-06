# panorama-view
Fast Panorama Stitching for Image Sequence

# Demonstraition

## Input images
<img src="https://github.com/maletsden/panorama-view/blob/main/test-images/mountain_1.jpg?raw=true" width=200/>
<img src="https://github.com/maletsden/panorama-view/blob/main/test-images/mountain_2.jpg?raw=true" width=200/>
<img src="https://github.com/maletsden/panorama-view/blob/main/test-images/mountain_3.jpg?raw=true" width=200/>
<img src="https://github.com/maletsden/panorama-view/blob/main/test-images/mountain_4.jpg?raw=true" width=200/>

## Generated panorama
<img src="https://github.com/maletsden/panorama-view/blob/main/results/mountain.jpg?raw=true" width=800/>


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