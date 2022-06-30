#pragma once


#include <vector>
#include <stdint.h>


std::vector<float> gen_matrix(const uint32_t& rows, const uint32_t& cols);
std::vector<float> gen_matrix(const uint32_t& batch, const uint32_t& rows, const uint32_t& cols);

void write_array(const std::vector<float>& vec, const char* namefile);
std::vector<float> read_array(const char* namefile, const uint32_t& size);



float* gen_filter_nchw(const uint32_t& num_in_channels = 3,
				   	   const uint32_t& num_out_channels = 3);



void transform_array_ones(const float* mat, const uint32_t& size, float* res);
void matrix_multiplication(const float* mat_1, const float* mat_2,
						   const uint32_t& r1, const uint32_t& c2, const uint32_t& d,
						   float* res);
void apply_bias_matrix(float* mat, const uint32_t& rows, const uint32_t& cols,
					   const float* bias);
void apply_gelu_matrix(float* mat, const uint32_t& size);
void print_array(const float* arr, const uint32_t& rows, const uint32_t& cols);
bool check_eq_matrices(const float* mat1, const float* mat2,
					   const uint32_t& rows, const uint32_t& cols,
					   const float eps);
