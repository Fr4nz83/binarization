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

	auto mat = gen_matrix(size_batch * image_channels, image_height, image_width); // Generate input matrix according to the NCHW format.
	float* img_data = mat.data();
	std::cout << "Size input: " << mat.size() * sizeof(float) << " bytes" << std::endl;



	//=============== Set up the layers to be used within the MLP-Mixer block =================

	// Generate the parameters to be used for the first batch-norm layer...
	auto bn1 = generate_random_bn("bn_1", image_width, image_height, image_channels);


	//=============== Set up the MLP-Mixer block =================

	MLPMixer mlp_mixer("mlpmix_1", image_height, image_width, image_channels,
					   bn1);



	//=============== MLP-Mixer execution =================

	// CUDA variables needed to measure the time the various operations take.
	cudaEvent_t start, end_load, stop;
	cudaEventCreate(&start); cudaEventCreate(&end_load), cudaEventCreate(&stop);


	// 1 - Load input data from CPU to GPU and allocate space for the output.
	cudaEventRecord(start);
	mlp_mixer.load_input_gpu(size_batch, {img_data});
	cudaEventRecord(end_load);


	// 2 - MLP-mixer block execution.
	// NOTE: we allocate 32 threads (1 warp) per block.
	mlp_mixer.execute_layer();
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
