cmake_minimum_required(VERSION 3.16)
project(panorama_view)

set(CMAKE_CXX_STANDARD 17)

# Build release version if not specified otherwise.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

FIND_PACKAGE(Eigen3 REQUIRED)
find_package(JPEG REQUIRED)

# ----- Panorama View ----- #
set(IMAGE_STITCHER_FILES
    src/image_stitcher/image_stitcher.h
    src/image_stitcher/homography_calculator/homography_calculator.cpp
    src/image_stitcher/homography_calculator/homography_calculator.h
    src/image_stitcher/image_transformation/image_transformation.cpp
    src/image_stitcher/image_transformation/image_transformation.h
    src/image_stitcher/knn_finder/knn_finder.cpp
    src/image_stitcher/knn_finder/knn_finder.h)
set(SOURCE_FILES
    src/jpeg_img.cpp
    src/jpeg_handler/jpeg_handler.cpp
    ${IMAGE_STITCHER_FILES})
add_executable(${PROJECT_NAME} main.cpp ${SOURCE_FILES})

# ----- threat warnings as errors ----- #
if(MSVC)
    target_compile_options(panorama_view PRIVATE /W4 /WX)
else()
    target_compile_options(panorama_view PRIVATE -Wall -Wextra -pedantic -Werror)
endif()

target_link_libraries(${PROJECT_NAME} JPEG::JPEG Eigen3::Eigen)

# TODO: remove opencv later
FIND_PACKAGE(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# ----- Google Tests ----- #
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(TESTS_SOURCE_FILES
    tests/main_test.cpp
    tests/test_homography_calculator.h
        tests/test_knn_finder.h
    tests/test_image_transformation.h
    ${IMAGE_STITCHER_FILES})
add_executable(panorama_view_tester ${TESTS_SOURCE_FILES})
target_link_libraries(panorama_view_tester ${GTEST_LIBRARIES} Threads::Threads Eigen3::Eigen)
