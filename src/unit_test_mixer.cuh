/*
 * unit_test_mixer.hpp
 *
 *  Created on: 12 sep 2022
 *      Author: lettich
 */
#pragma once


#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>

#include "dataset_reader.h"

#include "layers/layer_MLPMixer.cuh"
#include "layers/util_layers.cuh"

#include "generator.h"
#include "cnpy.h"



int test_mlp_mixer()
{
	std::cout << "*** MLP-Mixer block unit test *** " << std::endl;



    //=============== Device Configuration =================
	int dev = 0;
	cudaSetDevice(dev);



	//=============== Generate matrices =================

	constexpr uint32_t size_batch = 1,
					   image_height = 32,
			  	  	   image_width = 32,
					   image_channels = 3;

	auto mat1 = gen_matrix(size_batch * image_channels, image_height, image_width); // Generate input matrix according to the NCHW format.
	float* img1_data = mat1.data();
	std::cout << "Size input: " << mat1.size() * sizeof(float) << " bytes" << std::endl;



	//=============== Set up  layer =================

	// Generate the parameters of the layers within the MLP-Mixer block...


	// Instantiate the MLP-Mixer block...
	// MLPMixer mlp_mixer("mlpm_1", ...);



	//=============== Layer execution =================

	// CUDA variables needed to measure the time the various operations take.
	cudaEvent_t start, end_load, stop;
	cudaEventCreate(&start); cudaEventCreate(&end_load), cudaEventCreate(&stop);


	// 1 - Load input data from CPU to GPU and allocate space for the output.
	cudaEventRecord(start);
	//sum_l1.load_input_gpu(img1_data, img2_data);
	cudaEventRecord(end_load);


	// 2 - MLP-mixer block execution.
	// NOTE: we allocate 32 threads (1 warp) per block.
	//sum_l1.execute_layer();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);


	// 3 - Copy output from GPU to CPU.
	// float *test_output = new float[sum_l1.output_size()];
	// sum_l1.download_output_gpu(test_output);


	// Compute the execution time of the various steps.
	float ms_load, ms_kernel;
	cudaEventElapsedTime(&ms_load, start, end_load);
	cudaEventElapsedTime(&ms_kernel, end_load, stop);
	std::cout << "Load time: " << ms_load << " ms." << std::endl;
	std::cout << "Kernel execution time: " << ms_kernel << " ms." << std::endl;


	// Check for the GPU output correctness.


	// delete[] test_output;
	cudaEventDestroy(start);
	cudaEventDestroy(end_load);
	cudaEventDestroy(stop);


	return 0;
}
