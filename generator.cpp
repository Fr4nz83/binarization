#include "generator.h"
#include <random>
#include <stdio.h>


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
