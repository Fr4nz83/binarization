cmake_minimum_required(VERSION 3.9)
project(Binarization CUDA C CXX)



# *** CXX FLAGS ***
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)



# *** CUDA FLAGS ***
# NOTE: if CUDA was installed via a web installer, one should ensure that the environment variable CUDA_TOOLKIT_ROOT_DIR
# is set, e.g., CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-11.5/bin/", otherwise cmake will raise an error.
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)



if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

MESSAGE( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif ()


if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-missing-braces -Wno-unused-parameter")


    if(USE_SANITIZERS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    endif()


    if(RUNAWARE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRUNAWARE")
    endif()
endif()



# Pull in the libraries needed by the project.
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})
link_directories(${ZLIB_INCLUDE_DIRS})

find_package(JPEG REQUIRED)
include_directories(${JPEG_INCLUDE_DIRS})
link_directories(${JPEG_INCLUDE_DIRS})


# Compile the libraries needed by the project.
add_subdirectory(cnpy EXCLUDE_FROM_ALL)
include_directories(${PROJECT_SOURCE_DIR}/cnpy)


# Specify the directories where the targets are located.
add_subdirectory(src)
