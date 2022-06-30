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
	std::cout << "*** Convolution FP layer unit test *** " << std::endl;


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



	//================ Setup Network layer =================

	// *** Setup initial convolutional Layer *** //
	ConvLayer in_conv_layer("InConv",
							image_height,
							image_width,
							filter_height,
							filter_width,
							image_channels,
							num_filters);



	//================ Layer execution =================

	// Setup ConvLayer filters.
	in_conv_layer.initialize_filters(filter_test);

	// Load ConvLayer input data.
	in_conv_layer.load_input(size_batch, img_data);

	// Allocate space for output (and residuals, if needed).
	in_conv_layer.allocate_output_gpu();

	// Perform the last actions to prepare the layer for its execution.
	in_conv_layer.ready();

	// Execute the layer.
	in_conv_layer.execute_layer();



	//================ Resource deallocation =================

	delete[] img_data;
	delete[] filter_test;


	return 0;
}

/**
 * @brief Full precision batch normalization unit test.
 */
int test_bnfp_layer()
{
	std::cout << "*** Batch normalization FP layer unit test *** " << std::endl;



    //=============== Device Configuration =================
	int dev = 0;
	cudaSetDevice(dev);



	//=============== Read image dataset =================

	constexpr uint32_t size_batch = 10000,
					   image_height = 32,
			  	  	   image_width = 32,
					   image_channels = 10;

	// Generate a random matrix representing the image dataset.
	auto tmp = gen_matrix(size_batch * image_channels, image_height, image_width);
	float *img_data = tmp.data();
	std::cout << "Size input: " << tmp.size() * sizeof(float) << " bytes" << std::endl;



	//=============== Set up layer =================

	// Create scale and shift factors.
	float* scale_test = new float[image_channels];
	for(uint32_t i = 0; i < image_channels; i++) scale_test[i] = 1. / i;
	float* shift_test = new float[image_channels];
	for(uint32_t i = 0; i < image_channels; i++) shift_test[i] = i;


	// 1 - Instantiate the batch normalization layer.
	BatchNormFullPrecLayer bn_l1("bn_fp1",
								 image_width,    // Input width
								 image_height,   // Input height
								 image_channels, // Number of channels
								 scale_test, 	 // Pointer to the scale factors
								 shift_test); 	 // Pointer to the shift factors


	//=============== Kernel execution =================

	// CUDA variables needed to measure the time the various operations take.
	cudaEvent_t start, end_load, stop;
	cudaEventCreate(&start); cudaEventCreate(&end_load), cudaEventCreate(&stop);


	// 2 - Copy data from CPU to GPU.
	cudaEventRecord(start);
	bn_l1.load_input_gpu(img_data, size_batch);
	cudaEventRecord(end_load);

	// 3 - Prepare the layer for execution.
	BatchNormFullPrecLayer* gpu_copy = bn_l1.ready();

	// 4 - Batch normalization kernel execution.
	BNFPLayer <<<size_batch, 32>>>(gpu_copy);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// 5 - Retrieve the GPU output.
	float *test_output = new float[bn_l1.input_size()];
	bn_l1.download_output_gpu(test_output);


	// 6 - Compute the execution time of the various steps.
	float ms_load, ms_kernel;
	cudaEventElapsedTime(&ms_load, start, end_load);
	cudaEventElapsedTime(&ms_kernel, end_load, stop);
	std::cout << "Load time: " << ms_load << " ms." << std::endl;
	std::cout << "Kernel execution time: " << ms_kernel << " ms." << std::endl;


	// Verify GPU output correctness.
	bool check = true;
	constexpr float epsilon = 1e-5;
	for(uint32_t n = 0; n < size_batch; n++)
	{
		const uint32_t offset_img = n * image_channels * image_height * image_width;
		for(uint32_t c = 0; c < image_channels; c++)
		{
			const uint32_t offset_color = (image_height * image_width) * c;

			for(uint32_t i = 0; i < image_height * image_width; i++)
			{
				float cpu_o = scale_test[c] * img_data[offset_img + offset_color + i] + shift_test[c];

				// We have an error if the absolute difference between what's computed on CPU and that computed on GPU
				// is above a given epsilon.
				float gpu_o = test_output[offset_img + offset_color + i];
				if(std::abs(cpu_o - gpu_o) > epsilon) check = false;
					// std::cout << "ERRORE! " << cpu_o << " vs " << gpu_o << " (" << std::abs(cpu_o - gpu_o) << ") " << std::endl;
			}
		}
	}
	std::cout << "Check output GPU correctness: " << (check ? "OK" : "KO") << std::endl;



	delete[] scale_test;
	delete[] shift_test;
	delete[] test_output;
	cudaEventDestroy(start);
	cudaEventDestroy(end_load);
	cudaEventDestroy(stop);


	return 0;
}


int test_transpose_layer()
{
	std::cout << "*** FP matrix transposition layer unit test *** " << std::endl;



    //=============== Device Configuration =================
	int dev = 0;
	cudaSetDevice(dev);



	//=============== Generate image dataset =================

	constexpr uint32_t size_batch = 4000,
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


	delete[] test_output;
	cudaEventDestroy(start);
	cudaEventDestroy(end_load);
	cudaEventDestroy(stop);


	return 0;
}

/**
 * @brief Full precision batch normalization unit test.
 */
int test_bin_multi()
{
	std::cout << "*** Binary multiplication layer unit test *** " << std::endl;



    //=============== Device Configuration =================
	int dev = 0;
	cudaSetDevice(dev);



	//=============== Read image dataset =================

	constexpr uint32_t input_height = 100000,	// # of input entries.
					   weights_height = 4096,	// # of features
					   weights_width = 256;		// # Activation units

	// Generate a random matrix representing the image dataset.
	std::vector<float> tmp_img = gen_matrix(input_height, weights_height);
	float *img_data = tmp_img.data();
	std::cout << "Size input: " << tmp_img.size() * sizeof(float) << " bytes" << std::endl;



	//=============== Read weigths =================

	std::vector<float> tmp_w = gen_matrix(weights_height, weights_width);
	float *weights_data = tmp_w.data();
	std::cout << "Size weights: " << tmp_w.size() * sizeof(float) << " bytes" << std::endl;



	//=============== Read biases =================

	std::vector<float> tmp_b = gen_matrix(1, weights_width);
	float *bias_data = tmp_b.data();
	std::cout << "Size bias: " << tmp_b.size() * sizeof(float) << " bytes" << std::endl;



	//=============== Set up layer =================


	// 1 - Instantiate the batch normalization layer.
	std::cout << "Initializing the FC layer (with binary multiplication and FP output)..." << std::endl;
	BinaryMultiplicationLayer bm_l1("mb_1",
									weights_height, // Input features
									weights_width,  // Activation units
									weights_data,   // Pointer to the weights array
									bias_data); 	// Pointer to the bias vector



	//=============== Kernel execution =================

	// CUDA variables needed to measure the time the various operations take.
	cudaEvent_t start, end_load, end_mult, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&end_load),
	cudaEventCreate(&end_mult),
	cudaEventCreate(&stop);


	// 2 - Copy input data from CPU to GPU and allocate space for the final output.
	cudaEventRecord(start);
	std::cout << "Initializing the input and output of the layer..." << std::endl;
	bm_l1.load_input_gpu(img_data, input_height);
	bm_l1.allocate_output_gpu();
	cudaEventRecord(end_load);

	// 3 - Execute the layer, which is actually (1) input binarization, then (2) binary multiplication...
	bm_l1.execute();
	cudaEventRecord(end_mult);

	// 5 - Retrieve the GPU output.
	std::cout << "Size output: " << bm_l1.output_bytes() << " bytes" << std::endl;
	float *test_output = new float[bm_l1.output_size()];
	bm_l1.download_output_gpu(test_output);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);


	// 6 - Compute the execution time of the various steps.
	float ms_load, ms_kernel, ms_out;
	cudaEventElapsedTime(&ms_load, start, end_load);
	cudaEventElapsedTime(&ms_kernel, end_load, end_mult);
	cudaEventElapsedTime(&ms_out, end_mult, stop);
	std::cout << "Load time: " << ms_load << " ms." << std::endl;
	std::cout << "Input binarization + binary multi execution time: " << ms_kernel << " ms." << std::endl;
	std::cout << "Output to CPU time: " << ms_out << " ms." << std::endl;




	// *** Verify GPU output correctness *** //

	// Trasformazione in 1/-1 dell'input
	/*std::cout << "Trasformazione CPU 1/-1 matrice input" << std::endl;
	float *img_data_trans = new float[tmp_img.size()];
	transform_array_ones(img_data, tmp_img.size(), img_data_trans);

	/*std::cout << "Stampa matrice input originaria..." << std::endl;
	print_array(img_data, input_height, weights_height);

	std::cout << "Stampa matrice input 1/-1..." << std::endl;
	print_array(img_data_trans, input_height, weights_height);



	// Trasformazione in 1/-1 della matrice dei pesi.
	std::cout << "Trasformazione CPU 1/-1 matrice pesi" << std::endl;
	float *weights_trans = new float[tmp_w.size()];
	transform_array_ones(weights_data, tmp_w.size(), weights_trans);

	/*std::cout << "Stampa matrice pesi originaria..." << std::endl;
	print_array(weights_data, weights_height, weights_width);

	std::cout << "Stampa matrice pesi 1/-1..." << std::endl;
	print_array(weights_trans, weights_height, weights_width);


	// Calcolo moltiplicazione input x pesi con valori 1/-1.
	float *res_cpu = new float[input_height * weights_width];
	std::cout << "Calcolo moltiplicazione 1/-1 input x pesi su CPU" << std::endl;
	matrix_multiplication(img_data_trans, weights_trans,
						  input_height, weights_width, weights_height,
						  res_cpu);


	// std::cout << "Stampa risultato moltiplicazione 1/-1 su CPU" << std::endl;
	// print_array(res_cpu, input_height, weights_width);

	// std::cout << "Stampa risultato moltiplicazione 1/-1 su GPU" << std::endl;
	// print_array(test_output, input_height, weights_width);



	bool check = check_eq_matrices(res_cpu, test_output, input_height, weights_width, 1e-5);
	std::cout << "Check output GPU correctness: " << (check ? "OK" : "KO") << std::endl;

	delete[] img_data_trans;
	delete[] weights_trans;
	delete[] res_cpu;*/



	delete[] test_output;
	cudaEventDestroy(start);
	cudaEventDestroy(end_load);
	cudaEventDestroy(stop);


	return 0;
}
