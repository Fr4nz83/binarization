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
class BatchNormLayerFullPrec
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

	// ID layer.
	char name[8];



public:

	// *** CTORS / DTOR *** //

	ConvLayer(const char* name,
              unsigned input_height,
			  unsigned input_width,
			  unsigned filter_height,
			  unsigned filter_width,
			  unsigned input_channels,
			  unsigned output_channels,
			  unsigned batch,
			  unsigned stride_vertical = 1,
			  unsigned stride_horizontal = 1,
			  unsigned pad_h = 0,
			  unsigned pad_w = 0,
			  bool same_conv = true,
			  bool apply_bn = false,
			  bool save_residual = true)
	{
		strncpy(this->name, name, 8);

		this->input_height = input_height;
		this->input_width = input_width;
		this->filter_height = filter_height;
		this->filter_width = filter_width;
		this->input_channels = input_channels;
		this->output_channels = output_channels;
		this->batch = batch;
		this->stride_vertical = stride_vertical;
		this->stride_horizontal = stride_horizontal;
		this->apply_bn = apply_bn;
		this->same_conv = same_conv;


		// Calculate the padding required in case we are performing "same convolution".
		this->pad_h = this->same_conv ?
				      (( ( (input_height+stride_vertical-(input_height%stride_vertical)) / stride_vertical - 1) *
				    	stride_vertical+filter_height-input_height) >> 1) : pad_h;
		this->pad_w = this->same_conv ?
				      (( ( (input_width+stride_horizontal-(input_width%stride_horizontal)) / stride_horizontal - 1) *
				    	stride_horizontal+filter_width-input_width) >> 1) : pad_w;

		// Calculate the output size.
		this->output_height = this->same_conv ?
							  this->input_height :
							  (int)((((float)this->input_height +  2*this->pad_h - this->filter_height) / this->stride_vertical) + 1);
		this->output_width = this->same_conv ?
							 this->input_width :
							 (int)((((float)this->input_width +  2*this->pad_w - this->filter_width) / this->stride_horizontal) + 1);


		std::cout << "Input size: H=" << this->input_height << " W=" << this->input_width << " C=" << this->input_channels << std::endl;
		std::cout << "Same convolution? " << this->same_conv << " -- Calculated padding: H=" << this->pad_h << " W=" << this->pad_h << std::endl;
		std::cout << "Filters: N=" << this->output_channels << " H=" << this->filter_height << " W=" << this->filter_width << std::endl;
		std::cout << "Output size: H=" << this->output_height << " W=" << this->output_width << " C=" << this->output_channels << std::endl;

		// Initialize cuDNN.
		initialize_cuDNN();
	}

	~ConvLayer() {release();}



protected:

	// *** PROTECTED METHODS *** //

	bool initialize_cuDNN()
	{
		std::cout << "Initializing cuDNN!" << std::endl;

		cudnnStatus_t status;
		cudnnHandle_t cudnn;


		// Allocate cuDNN handle
		status = cudnnCreate(&cudnn);
	    if (status != CUDNN_STATUS_SUCCESS) return false;
	    std::cout << "cuDNN handle OK!" << std::endl;


	    // Allocate input tensor data structures.
	    // TODO: da settare il numero di immagini!
	    cudnnTensorDescriptor_t input_descriptor;
	    cudnnCreateTensorDescriptor(&input_descriptor);
	    status = cudnnSetTensor4dDescriptor(input_descriptor,
	                                        /*format=*/CUDNN_TENSOR_NCHW,
											/*dataType=*/CUDNN_DATA_FLOAT,
											/*batch_size=*/1,
											/*channels=*/this->input_channels,
											/*image_height=*/this->input_height,
											/*image_width=*/this->input_width);
	    std::cout << "cuDNN input tensor allocation OK!" << std::endl;


	    // Allocate output tensor data structures.
	    // TODO: da settare il numero di immagini!
	    cudnnTensorDescriptor_t output_descriptor;
	    cudnnCreateTensorDescriptor(&output_descriptor);
	    status = cudnnSetTensor4dDescriptor(output_descriptor,
	                                          /*format=*/CUDNN_TENSOR_NCHW,
	                                          /*dataType=*/CUDNN_DATA_FLOAT,
	                                          /*batch_size=*/1,
	                                          /*channels=*/this->output_channels,
	                                          /*image_height=*/this->output_height,
	                                          /*image_width=*/this->output_width);
	    std::cout << "cuDNN output tensor allocation OK!" << std::endl;


	    // Allocate kernel tensor data structures.
	    cudnnFilterDescriptor_t kernel_descriptor;
	    cudnnCreateFilterDescriptor(&kernel_descriptor);
	    status = cudnnSetFilter4dDescriptor(kernel_descriptor,
										    /*dataType=*/CUDNN_DATA_FLOAT,
											/*format=*/CUDNN_TENSOR_NCHW,
											/*out_channels=*/this->output_channels,
											/*in_channels=*/this->input_channels,
											/*kernel_height=*/this->filter_height,
											/*kernel_width=*/this->filter_width);
	    std::cout << "cuDNN kernel tensor allocation OK!" << std::endl;



	    // TODO: to be continued.


		return true;
	}

	ConvLayer* ready()
	{
		// Pointers sanity check.
		if (this->input_gpu == NULL)
		{
			fprintf(stderr, "Input data has not been uploaded to GPU.\n");
			exit(1);
		}
		if (this->output_gpu == NULL)
		{
			fprintf(stderr, "Output on GPU has not been allocated.\n");
			exit(1);
		}
		if (this->save_residual && this->save_residual_gpu == NULL)
		{
			fprintf(stderr, "Residual for saving on GPU has not been allocated.\n");
			exit(1);
		}

		// Allocate instance shadow pointer on GPU.
		CUDA_SAFE_CALL( cudaMalloc((void**)&(this->gpu), sizeof(ConvLayer)) );
		CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(ConvLayer), cudaMemcpyHostToDevice));
		return this->gpu;
	}

	void release()
	{
		if (this->output_gpu!=NULL)
		{
			CUDA_SAFE_CALL( cudaFree(this->output_gpu) );
			this->output_gpu=NULL;
		}
		if (this->filter_gpu != NULL)
		{
			CUDA_SAFE_CALL( cudaFree(this->filter_gpu) );
			this->filter_gpu = NULL;
		}
		if (this->bn_gpu != NULL)
		{
			CUDA_SAFE_CALL( cudaFree(this->bn_gpu) );
			this->bn_gpu = NULL;
		}
		if (this->gpu != NULL)
		{
			CUDA_SAFE_CALL( cudaFree(this->gpu) );
			this->gpu = NULL;
		}
		if (this->save_residual && this->save_residual_gpu != NULL)
		{
			CUDA_SAFE_CALL( cudaFree(this->save_residual_gpu) );
			this->save_residual_gpu=NULL;
		}
	}



public:

	// *** PUBLIC METHODS *** //

	void set_input_gpu(float* input_gpu) {this->input_gpu = input_gpu;}
	int input_size() {return input_channels*input_height*input_width*batch;}
	int input_bytes() {return input_size() * sizeof(float);}
	int filter_size() {return output_channels*input_channels*filter_height*filter_width;}
	int filter_bytes() {return filter_size() * sizeof(float);}
	int output_size() {return output_channels*output_height*output_width*batch;}
	int output_bytes() {return output_size() * sizeof(unsigned);}
	int bn_size() {return output_channels;}
	int bn_bytes() {return bn_size() * sizeof(float);}

	float* get_output_gpu() {return this->output_gpu;}

	float* get_residual_gpu(){return this->save_residual_gpu;}



	void initialize(const float* img_data, const float* filters, const float* bn = NULL)
	{
		// Read and allocate filters data.
		CUDA_SAFE_CALL(cudaMalloc((void**)&(this->filter_gpu), filter_bytes()));
		CUDA_SAFE_CALL(cudaMemcpy(this->filter_gpu, filters, filter_bytes(), cudaMemcpyHostToDevice));


		// Read and allocate the variables for batch normalization.
		if(this->apply_bn)
		{
			CUDA_SAFE_CALL(cudaMalloc((void**)&(this->bn_gpu), bn_bytes()));
			CUDA_SAFE_CALL(cudaMemcpy(this->bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice));
		}
		else this->bn_gpu = NULL;


		// Allocate output gpu.
		CUDA_SAFE_CALL(cudaMalloc((void**)&(this->output_gpu), output_bytes()));
		CUDA_SAFE_CALL(cudaMemset(this->output_gpu, 0, output_bytes()));


		// If required, allocate residual data structures for skip connections.
		if (this->save_residual)
		{
			CUDA_SAFE_CALL(cudaMalloc((void**)&(this->save_residual_gpu), output_bytes()));
			CUDA_SAFE_CALL(cudaMemset(this->save_residual_gpu, 0, output_bytes()) );
		}
	}

	ConvLayer* load_input(const float* img_data, const unsigned batch)
	{
		// Save the number of images in the dataset.
		this->batch = batch;

		// Copy input data to GPU.
		CUDA_SAFE_CALL(cudaMalloc((void**)&(this->input_gpu), this->input_bytes()));
		CUDA_SAFE_CALL(cudaMemcpy(this->input_gpu, img_data, this->input_bytes(), cudaMemcpyHostToDevice));

		return this->ready();
	}
};
