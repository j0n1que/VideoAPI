cmake_minimum_required(VERSION 3.22.1)

project(LibInfer C CXX)


set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)





include(FetchContent)


find_package(CUDAToolkit 12.1 REQUIRED)


message(STATUS "CUDA toolkit version: ${CUDA_VERSION}")


add_library(libinfer SHARED
  src/api.cpp
  src/bbox.cpp
  src/cuda.cpp
  src/infer.cpp
  src/logger.cpp
  src/model_loader.cpp
)


set_target_properties(libinfer PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)


target_link_libraries(libinfer PRIVATE
  C:/Users/j0n1que/TensorRT-10.0.0.6/lib/nvinfer.lib
  C:/Users/j0n1que/TensorRT-10.0.0.6/lib/nvonnxparser.lib
  CUDA::toolkit
  CUDA::cuda_driver
)


set_target_properties(libinfer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
