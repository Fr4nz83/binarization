#pragma once


#include <vector>
#include <stdint.h>


std::vector<float> gen_matrix(const uint32_t& rows, const uint32_t& cols);
std::vector<float> gen_matrix(const uint32_t& batch, const uint32_t& rows, const uint32_t& cols);

void write_array(const std::vector<float>& vec, const char* namefile);
std::vector<float> read_array(const char* namefile, const uint32_t& size);

float* gen_filter_nchw(const uint32_t& num_in_channels = 3,
				   	   const uint32_t& num_out_channels = 3);


