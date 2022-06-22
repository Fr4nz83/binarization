/*
 * unit_test.hpp
 *
 *  Created on: 19 giu 2022
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

#include <cooperative_groups.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "utility.h"
#include "sbnn32_param.h"
#include "sbnn32.cuh"
#include "data.h"
#include "dataset_reader.h"

#include "binarization.cuh"
#include "my_layers.cuh"

#include "generator.h"
#include "cnpy.h"



/**
 * @brief Full precision convolution layer test unit.
 */
int test_convfp_layer()
{
    using namespace cooperative_groups;

    //=============== Device Configuration =================
	int dev = 0;
	cudaSetDevice(dev);



	//=============== Read image dataset =================

	constexpr uint32_t image_height = 32,
			  image_width = 32,
			  image_channels = 3,
			  filter_height = 3,
			  filter_width = 3,
			  num_filters = 10;

	// Read the image dataset.
	std::string cifar10_dir = "../dataset/data_batch_1.bin";
	auto set_images = ImgDatasetReader<image_height,image_width>::read_dataset_cifar10_float(cifar10_dir);
	std::cout << "Number of images: " << set_images.size() << std::endl;
	const uint32_t size_batch = set_images.size();

	// Convert the dataset into a NCHW float array.
	float *img_data = ImgDatasetReader<image_height,image_width>::transform_dataset_nchw_float(set_images);
	float *filter_test = gen_filter_nchw(image_channels, num_filters);


	// const uint32_t batch = set_images.size();
	// uint32_t *images_test = (uint32_t*) malloc(batch * image_height * image_width * sizeof(uint32_t));
	// uint32_t *image_labels_test = (uint32_t*) malloc(batch * sizeof(uint32_t));
	// read_CIFAR10_raw(cifar10_dir, images_test, image_labels_test, batch);



	//================ Setup Network layers =================

	// *** Setup initial convolutional Layer *** //
	ConvLayer in_conv_layer = ConvLayer("InConv",
										image_height,
										image_width,
										filter_height,
										filter_width,
										image_channels,
										num_filters);

	// Setup ConvLayer filters.
	in_conv_layer.initialize_filters(filter_test);

	// Load ConvLayer input data.
	in_conv_layer.load_input(size_batch, img_data);

	// Execute ConvLayer.
	in_conv_layer.execute_layer();
}

/**
 * @brief Full precision batch normalization unit test.
 */
int test_bnfp_layer()
{
    using namespace cooperative_groups;

    //=============== Device Configuration =================
	int dev = 0;
	cudaSetDevice(dev);



	//=============== Read image dataset =================

	constexpr uint32_t image_height = 32,
			  	  	   image_width = 32,
					   image_channels = 3;

	// Read the image dataset.
	std::string cifar10_dir = "../dataset/data_batch_1.bin";
	auto set_images = ImgDatasetReader<image_height,image_width>::read_dataset_cifar10_float(cifar10_dir);
	std::cout << "Number of images: " << set_images.size() << std::endl;
	const uint32_t size_batch = set_images.size();

	// Convert the dataset into a NCHW float array.
	float *img_data = ImgDatasetReader<image_height,image_width>::transform_dataset_nchw_float(set_images);
	// auto tmp = gen_matrix(size_batch * image_channels, image_height, image_width);
	// float *img_data = tmp.data();


	// Create scale and shift factors.
	float* scale_test = new float[image_channels];
	for(uint32_t i = 0; i < image_channels; i++) scale_test[i] = 1. / i;
	float* shift_test = new float[image_channels];
	for(uint32_t i = 0; i < image_channels; i++) shift_test[i] = i;


	BatchNormFullPrecLayer bn_l1("bn_fp1",
								 image_width,    // Input width
								 image_height,   // Input height
								 image_channels, // Number of channels
								 scale_test, 	 // Pointer to the scale factors
								 shift_test); 	 // Pointer to the shift factors
	BatchNormFullPrecLayer* gpu_copy = bn_l1.load_input_gpu(img_data, size_batch);


	// Batch normalization kernel execution.
	// - One block per image.
	// - 32 threads (1 warp) per block
	BNFPLayer <<<500, 32>>>(gpu_copy);


	float *test_output = new float[bn_l1.input_size()];
	bn_l1.download_output_gpu(test_output);


	// Veify GPU output correctness.
	for(uint32_t n = 0; n < size_batch; n++)
	{
		const uint32_t offset_img = n * image_channels * image_height * image_width;
		for(uint32_t c = 0; c < image_channels; c++)
		{
			const uint32_t offset_color = (image_height * image_width) * c;

			for(uint32_t i = 0; i < image_height * image_width; i++)
			{
				float cpu_o = scale_test[c] * img_data[offset_img + offset_color + i] + shift_test[c];
				float gpu_o = test_output[offset_img + offset_color + i];
				if(std::abs(cpu_o - gpu_o) > 1e-5)
					std::cout << "ERRORE! " << cpu_o << " vs " << gpu_o << " (" << std::abs(cpu_o - gpu_o) << ") " << std::endl;
			}
		}
	}

	return 0;
}


int test_transpose_layer()
{
    using namespace cooperative_groups;

    //=============== Device Configuration =================
	int dev = 0;
	cudaSetDevice(dev);



	//=============== Generate fake image dataset =================

	constexpr uint32_t size_batch = 45000,
					   image_height = 32,
			  	  	   image_width = 32,
					   image_channels = 10,
					   THREADS_PER_BLOCK = 32;
	auto tmp = gen_matrix(size_batch * image_channels, image_height, image_width);
	float* img_data = tmp.data();
	std::cout << "Size input: " << tmp.size() * sizeof(float) << " bytes" << std::endl;


	//=============== Set up layer =================

	TransposeFullPrecLayer tr_l1("tr_fp1",
								 image_width,     // Input width
								 image_height,    // Input height
								 image_channels); // Number of channels;



	//=============== Layer execution =================

	// CUDA variables needed to measure the time the various operations take.
	cudaEvent_t start, end_load, stop;
	cudaEventCreate(&start); cudaEventCreate(&end_load), cudaEventCreate(&stop);


	// 1 - Load input data from CPU to GPU.
	cudaEventRecord(start);
	tr_l1.load_input_gpu(img_data, size_batch);
	cudaEventRecord(end_load);

	// 2 - Allocate output memory on GPU.
	tr_l1.allocate_output_gpu();

	// 3 - Prepare the layer for execution
	TransposeFullPrecLayer* gpu_copy = tr_l1.ready();

	// 4 - Transpose kernel execution.
	// NOTE: we allocate 32 threads (1 warp) per block.
	TransposeFPLayer <<<size_batch, THREADS_PER_BLOCK>>> (gpu_copy);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// 5- Copy output from GPU to CPU.
	float *test_output = new float[tr_l1.input_size()];
	tr_l1.download_output_gpu(test_output);


	// 6 - Compute the execution time of the various steps.
	float ms_load, ms_kernel;
	cudaEventElapsedTime(&ms_load, start, end_load);
	cudaEventElapsedTime(&ms_kernel, end_load, stop);
	std::cout << "Load time: " << ms_load << " ms." << std::endl;
	std::cout << "Kernel execution time: " << ms_kernel << " ms." << std::endl;



	// Verify the GPU output correctness.
	bool check = true;
	for(uint32_t n = 0; n < size_batch; n++)
	{
		//std::cout << "Image " << n << std::endl;

		const uint32_t offset_img = n * image_channels * image_height * image_width;
		for(uint32_t c = 0; c < image_channels; c++)
		{
			//std::cout << "Channel " << c << std::endl;

			const uint32_t offset_color = (image_height * image_width) * c;
			for(uint32_t row = 0; row < image_height; row++)
			{
				//std::cout << "Row " << row << ": ";

				for(uint32_t column = 0; column < image_width; column++)
				{
					float cpu_o = img_data[offset_img + offset_color + (row * image_width) + column];
					float gpu_o = test_output[offset_img + offset_color + (column * image_height) + row];

					if(cpu_o != gpu_o) check = false; // std::cout << "ERRORE!" << std::endl;
				}
				// std::cout << std::endl;
			}
		}
	}
	std::cout << "Check output GPU correctness: " << (check ? "OK" : "KO") << std::endl;


	return 0;
}
