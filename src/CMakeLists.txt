add_executable(bin bin.cu data.cpp generator.cpp)

target_link_libraries(bin cnpy jpeg z cublas cudnn ${Boost_LIBRARIES})
