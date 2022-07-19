#include "generator.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <math.h>



//================ Macro Definition ===============
#define BITWIDTH 32
#define LOG_BITWIDTH 5
#define CEIL(X) (((X) + BITWIDTH - 1) >> LOG_BITWIDTH) // Equivalente a ceil(X/32)
#define FEIL(X) ((((X) + BITWIDTH - 1) >> LOG_BITWIDTH) << LOG_BITWIDTH) // Equivalente a ceil(X/32)*32 (ovvero, il multiplo di 32 >= X)



std::vector<float> gen_matrix(const uint32_t& rows, const uint32_t& cols)
{
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> float_dist(-5, 5);
	    
	std::vector<float> vec(rows * cols);
	for (auto& el : vec) el = (float) float_dist(gen);
	    
	return vec;
}

std::vector<float> gen_matrix(const uint32_t& batch, const uint32_t& rows, const uint32_t& cols)
{   
    	return gen_matrix(batch * rows, cols);
}

void write_array(const std::vector<float>& vec, const char* namefile)
{
	auto fp = fopen(namefile, "w");
	// check for error here

	for(const auto& el : vec) 
		fprintf(fp, "%f ", el);

	fclose(fp);
}

std::vector<float> read_array(const char* namefile, const uint32_t& size)
{   
    std::vector<float> array(size);
    
    auto cf = fopen(namefile, "r");
    for (int i=0; i < size; i++)
    {
    	fscanf(cf, "%f", &array[i]);
    }   
    fclose(cf);
    
    return(array);
}

float* gen_filter_nchw(const uint32_t& num_in_channels,
				   	   const uint32_t& num_out_channels)
{
	constexpr uint32_t size_filter = 3;

	// Filtro di test.
	const float kernel_template[size_filter][size_filter] =
	{
	  {1,  1, 1},
	  {1, -8, 1},
	  {1,  1, 1}
	};

	float *kernel_ptr = new float[num_out_channels * num_in_channels * size_filter * size_filter];
	for (int o = 0; o < num_out_channels; o++)
	{
		const uint32_t offset_filter = o * (num_in_channels * size_filter * size_filter);
		for (int i = 0; i < num_in_channels; i++)
		{
			const uint32_t offset_in_channel = offset_filter + i * (size_filter * size_filter);
			for (int row = 0; row < size_filter; row++)
			{
				const uint32_t offset_row = offset_in_channel + row * size_filter;
				for (int column = 0; column < size_filter; column++)
					kernel_ptr[offset_row + column] = kernel_template[row][column];
			}
		}
	}

	return(kernel_ptr);
}

void transform_array_ones(const float* mat, const uint32_t& size, float* res)
{
	for(uint32_t i = 0; i < size; i++) res[i] = mat[i] >= 0 ? 1 : -1;
}

void matrix_multiplication(const float* mat_1, const float* mat_2,
						   const uint32_t& r1, const uint32_t& c2, const uint32_t& d,
						   float* res)
{
	for (uint32_t i = 0; i < r1; i++)
		for (uint32_t j = 0; j < c2; j++)
		{
			res[i * c2 + j] = 0;
			for (uint32_t k = 0; k < d; k++)
				res[i * c2 + j] += mat_1[i * d + k] * mat_2[k * c2 + j];
		}
}

void apply_bias_matrix(float* mat, const uint32_t& rows, const uint32_t& cols,
					   const float* bias)
{
	for(uint32_t row = 0; row < rows; row++)
		for(uint32_t col = 0; col < cols; col++)
			mat[row * cols + col] += bias[col];
}

void apply_gelu_matrix(float* mat, const uint32_t& size)
{
	for(uint32_t i = 0; i < size; i++)
		mat[i] = (0.5 * mat[i]) * (1 + tanhf( sqrtf(2/M_PI) * (mat[i] + 0.044715 * powf(mat[i], 3)) ));
}

void print_array(const float* arr, const uint32_t& rows, const uint32_t& cols)
{
	for(uint32_t row = 0; row < rows; row++)
	{
		std::cout << "Line " << row << ": ";
		for(uint32_t col = 0; col < cols; col++)
		{
			std::cout << arr[row * cols + col] << " ";
		}
		std::cout << std::endl;
	}
}

void print_binarized_array(const unsigned* arr, const uint32_t& rows, const uint32_t& cols)
{
	// NOTE: here we assume that each block contains 32 32-bit rows, and that the blocks
	// 		 are arranged according to the column-major layout, i.e., we have "#CEIL(cols)" binarized stripes
	// 	     of "num_rows x 32 cols" bit matrices arranged consecutively.

	const int gdx = CEIL(rows); // Height of the binarized output matrix.
	const int gdy = CEIL(cols); // Width of the binarized output matrix.

	for(uint32_t row = 0; row < rows; row++)
	{
		std::cout << "Line " << row << ": ";
		for(uint32_t col = 0; col < cols; col++)
		{
			// Compute the memory offset of the bit containing the element (bit) of the output we want to print.
			// unsigned* ptr = &arr[by*(gdx*32) + bx*32];
			const unsigned by = col / 32;
			unsigned val = arr[by*(gdx*32) + row];

			// Now, determine the value of the bit (while keeping in mind that the bits have been reversed in each word),
			// and then translate it to the corresponding float value.
			const unsigned mask = ((unsigned) 1) << (31 - (col % 32));
			const float res = val & mask ? 1. : -1.;

			std::cout << res << " ";
		}
		std::cout << std::endl;
	}
}

void transpose_matrix(const float* src, float* dest, const uint32_t& rows, const uint32_t& cols)
{
	for(uint32_t r = 0; r < rows; r++)
		for(uint32_t c = 0; c < cols; c++)
			dest[c * rows + r] = src[r * cols + c];
}

bool check_eq_matrices(const float* mat1, const float* mat2,
					   const uint32_t& rows, const uint32_t& cols,
					   const float eps)
{
	bool check = true;
	for(uint32_t row = 0; row < rows; row++)
	{
		for(uint32_t col = 0; col < cols; col++)
			if(std::abs(mat1[row * cols + col] - mat2[row * cols + col]) > eps)
			{
				check = false;
				break;
			}
	}

	return check;
}

bool check_eq_matrices_binarized(const float* mat1, const unsigned* mat2,
					   	   	     const uint32_t& rows, const uint32_t& cols)
{
	// NOTE: here we assume that each block contains 32 32-bit rows, and that the blocks
	// 		 are arranged according to the column-major layout, i.e., we have "#CEIL(cols)" binarized stripes
	// 	     of "num_rows x 32 cols" bit matrices arranged consecutively.

	const int gdx = CEIL(rows); // Height of the binarized output matrix.
	const int gdy = CEIL(cols); // Width of the binarized output matrix.


	bool check = true;
	for(uint32_t row = 0; row < rows; row++)
	{
		for(uint32_t col = 0; col < cols; col++)
		{
			// Compute the memory offset of the bit containing the element (bit) of the output we want to print.
			const unsigned by = col / 32;
			unsigned val = mat2[by*(gdx*32) + row];

			// Now, determine the value of the bit (while keeping in mind that the bits have been reversed in each word),
			// and then translate it to the corresponding float value.
			const unsigned mask = ((unsigned) 1) << (31 - (col % 32));
			const float res = val & mask ? 1. : -1.;

			if(mat1[row * cols + col] != res)
			{
				check = false;
				break;
			}
		}
	}

	return check;
}

bool verify_padding_binarized_matrix(const unsigned* mat, const uint32_t& rows, const uint32_t& cols)
{
	// NOTE: here we assume that each block contains 32 32-bit rows, and that the blocks
	// 		 are arranged according to the column-major layout, i.e., we have "#CEIL(cols)" binarized stripes
	// 	     of "num_rows x 32 cols" bit matrices arranged consecutively.

	const int gdx = CEIL(rows); // Height of the binarized output matrix.
	const int gdy = CEIL(cols); // Width of the binarized output matrix.

	std::cout << "Boundary rows: " << FEIL(rows) << " (" << rows << ")" << std::endl;
	std::cout << "Boundary columns: " << FEIL(cols) << " (" << cols << ")" << std::endl;

	bool check = true;
	for(uint32_t row = rows; row < FEIL(rows); row++)
	{
		for(uint32_t col = cols; col < FEIL(cols); col++)
		{
			// Compute the memory offset of the bit containing the element (bit) of the output we want to print.
			const unsigned by = col / 32;
			unsigned val = mat[by*(gdx*32) + row];

			// Now, determine the value of the bit (while keeping in mind that the bits have been reversed in each word),
			// and then translate it to the corresponding float value.
			const unsigned mask = ((unsigned) 1) << (31 - (col % 32));
			const float res = val & mask ? 1. : -1.;

			if(res != -1.)
			{
				check = false;
				break;
			}
		}
	}

	return check;
}
