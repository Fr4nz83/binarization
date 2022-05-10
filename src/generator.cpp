#include "generator.h"
#include <random>
#include <stdio.h>


std::vector<float> gen_matrix(const uint32_t& rows, const uint32_t& cols, const float& lb, const float& ub)
{
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> float_dist(lb, ub);
	    
	std::vector<float> vec(rows * cols);
	for (auto& el : vec) el = (float) float_dist(gen);
	    
	return vec;
}

std::vector<float> gen_matrix(const uint32_t& batch, const uint32_t& rows, const uint32_t& cols, const float& lb, const float& ub)
{   
    	return gen_matrix(batch * rows, cols, lb, ub);
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
