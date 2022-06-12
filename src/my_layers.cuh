/*
 * my_layers.cuh
 *
 *  Created on: 9 giu 2022
 *      Author: francesco lettich
 */

#pragma once


// *** INCLUDES *** //

#include <cudnn.h>
#include "sbnn32_param.h"



/** @brief This class implements a batch normalization layer which can operate on different channels.
 *
 *  @note The implementation assumes that a dataset of images follows the NCHW format.
 */
class BatchNormFullPrecLayer
{
public:

	// Layer input data structures
	float* input_gpu;
	unsigned input_width;
	unsigned input_height;
	unsigned input_channels;

	// Layer output data structures
	float* output_gpu;
	unsigned output_width;
	unsigned output_height;
	unsigned output_channels;

	// GPU shadow.
	// BatchNormLayerFullPrec* gpu;
};



/** @brief This class implements a full-precision convolutional layer based on cuDNN.
 *
 *  @note The implementation assumes that a dataset of images follows the NCHW format, while the set filters follows the HWIO format.
 */
class ConvLayer
{
protected:

	// *** FIELDS *** //

	// Input
	float* input_gpu;
	unsigned input_width;
	unsigned input_height;
	unsigned input_channels;

	// Filters
	float* filter_gpu;
	unsigned filter_width;
	unsigned filter_height;
	unsigned filter_channels;

	// Output
	float* output_gpu;
	unsigned output_width;
	unsigned output_height;
	unsigned output_channels;

	// Batch normalization
	float* bn_gpu;

	// Convolution general properties.
	bool apply_bn;
	bool same_conv;
	unsigned batch;
	unsigned stride_vertical;
	unsigned stride_horizontal;
	unsigned pad_h;
	unsigned pad_w;

	// Skip connections.
	bool save_residual;
	float *save_residual_gpu;

	// GPU shadow.
	ConvLayer* gpu;

	// cuDNN data structures.
	cudnnHandle_t cudnn;
	cudnnTensorDescriptor_t input_descriptor;
	cudnnTensorDescriptor_t output_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    size_t workspace_bytes;
    void *d_workspace;

	// ID layer.
	char name[8];



	// *** PROTECTED METHODS *** //

	bool initialize_cuDNN();
	bool ready();
	void release();
	void release_cuDNN();



public:

	// *** PUBLIC CTORS / DTOR *** //

	ConvLayer(const char* name,
              unsigned input_height,
			  unsigned input_width,
			  unsigned filter_height,
			  unsigned filter_width,
			  unsigned input_channels,
			  unsigned output_channels,
			  unsigned stride_vertical = 1,
			  unsigned stride_horizontal = 1,
			  unsigned pad_h = 0,
			  unsigned pad_w = 0,
			  bool same_conv = true,
			  bool apply_bn = false,
			  bool save_residual = true);
	~ConvLayer()
	{
		release();
		release_cuDNN();
	}



	// *** PUBLIC METHODS *** //

	int input_size() {return input_channels*input_height*input_width*batch;}
	int input_bytes() {return input_size() * sizeof(float);}
	int filter_size() {return output_channels*input_channels*filter_height*filter_width;}
	int filter_bytes() {return filter_size() * sizeof(float);}
	int output_size() {return output_channels*output_height*output_width*batch;}
	int output_bytes() {return output_size() * sizeof(unsigned);}
	int bn_size() {return output_channels;}
	int bn_bytes() {return bn_size() * sizeof(float);}

	// Sets and gets for input/output pointers.
	void set_input_gpu(float* input_gpu) {this->input_gpu = input_gpu;}
	float* get_output_gpu() {return this->output_gpu;}
	float* get_residual_gpu(){return this->save_residual_gpu;}

	bool initialize_filters(const float* filters, const float* bn = NULL);
	bool load_input(const unsigned& batch_size, const float* img_data);
	bool execute_layer();
};


// Pull in the definitions of the methods associated with the class ConvLayer.
#include "my_layers_conv.inl"
