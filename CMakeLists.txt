cmake_minimum_required(VERSION 3.22.1)

project(LibInfer
	C CXX)

include(FetchContent)

find_package(CUDAToolkit 12.1 REQUIRED)

add_library(infer SHARED
	src/api.cpp
	src/bbox.cpp
	src/cuda.cpp
	src/infer.cpp
	src/logger.cpp
	src/model_loader.cpp)

set_target_properties(infer PROPERTIES
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED ON)

target_link_libraries(infer PRIVATE
	nvinfer
	nvonnxparser
	CUDA::toolkit
	CUDA::cuda_driver)
