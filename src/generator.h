#pragma once


#include <vector>
#include <stdint.h>


std::vector<float> gen_matrix(const uint32_t& rows, const uint32_t& cols, const float& lb, const float& ub);
std::vector<float> gen_matrix(const uint32_t& batch, const uint32_t& rows, const uint32_t& cols, const float& lb, const float& ub);

void write_array(const std::vector<float>& vec, const char* namefile);
std::vector<float> read_array(const char* namefile, const uint32_t& size);
