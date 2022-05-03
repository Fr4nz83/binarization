add_executable(bmm bmm.cu data.cpp generator.cpp)
add_executable(bin bin.cu data.cpp generator.cpp)

target_link_libraries(bmm cnpy jpeg z cublas ${Boost_LIBRARIES})
target_link_libraries(bin cnpy jpeg z cublas ${Boost_LIBRARIES})
