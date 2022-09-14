/*
 * MLPMixer.h
 *
 *  Created on: 1 ago 2022
 *      Author: lettich
 */
#pragma once


/*** INCLUDES ***/

#include "layer_batch_norm.cuh"
#include "layer_binmult.cuh"
#include "layer_transpose.cuh"
#include "layer_matrix_sum.cuh"



class MLPMixer : public Layer
{
protected:

	// *** FIELDS *** //

	unsigned input_size_batch;

	// Input fields.
	float* input_gpu;			// Memory allocated for FP input.
	unsigned input_height;
	unsigned input_width;
	unsigned input_channels;


	// Output fields.
	float* output_gpu;			// Memory allocated for FP output.


	// Residuals used within the MLP-Mixer block.
	float* gpu_residuals_1;
	float* gpu_residuals_2;


	// Layers used within the MLP-Mixer block.
	BatchNormFullPrecLayer* bn1_layer;
	BinaryMultiplicationLayer* bmm1_layer; // Must binarize a FP input and produce a binarized output.
	BinaryMultiplicationLayer* bmm2_layer; // Takes an already binarized input and must produce a transposed FP output.

	BatchNormFullPrecLayer* bn2_layer;
	BinaryMultiplicationLayer* bmm3_layer; // Must binarize a FP input and produce a binarized output.
	BinaryMultiplicationLayer* bmm4_layer; // Takes an already binarized input and must produce a transposed FP output.

	TransposeFullPrecLayer* tr_layer; // NOTE: This layer is potentially used twice in the MLP-Mixer block.
	MatrixSumLayer* sum_layer; // NOTE: This layer is potentially used twice in the MLP-Mixer block.

	char name[8];


public:

	// *** PUBLIC CTORS / DTOR *** //

	MLPMixer(const char* name,
			 const unsigned& size_batch,
			 const unsigned& data_width,
			 const unsigned& data_channels,
			 const BatchNormFullPrecLayer::BatchNormLayerParams& params_bn1);

	virtual ~MLPMixer(){this->release();};



	// *** PUBLIC METHODS *** //

	inline virtual void release();
	inline virtual MLPMixer* ready();

	inline virtual int get_size_batch() {return this->input_size_batch;}

	inline virtual int input_size() {return this->input_size_batch * this->input_height * this->input_width * this->input_channels;}
	inline virtual int input_bytes() {return this->input_size() * sizeof(float);}
	inline virtual int get_input_width() {return this->input_width;}
	inline virtual int get_input_heigth() {return this->input_height;}
	inline virtual int get_input_channels() {return this->input_channels;}

	inline virtual int output_size() {return this->input_size();}
	inline virtual int output_bytes() {return this->input_bytes();}
	inline virtual int get_output_width() {return this->get_input_width();}
	inline virtual int get_output_height() {return this->get_input_heigth();}
	inline virtual int get_output_channels() {return this->get_input_channels();};
	inline virtual int get_output_size_batch() {return this->get_size_batch();};

	// TODO: da implementare.
	inline virtual void allocate_output_gpu() {};
	inline virtual void load_input_gpu(const unsigned& size_batch, const std::vector<void*>& input) {};
	inline virtual void set_input_gpu(const unsigned& size_batch, const std::vector<void*>& input_gpu) {};

	// TODO: da implementare.
	inline virtual void* get_output_gpu() {return 0;};
	inline virtual void download_output_gpu(void* output) {};

	// TODO: da implementare.
	inline virtual void execute_layer();
};



MLPMixer::MLPMixer(const char* name,
				   const unsigned& size_batch,
				   const unsigned& width,
				   const unsigned& channels,
				   const BatchNormFullPrecLayer::BatchNormLayerParams params_bn1) :

input_size_batch(size_batch),
input_gpu(NULL),
input_height(0),
input_width(width),
input_channels(channels),
output_gpu(NULL),
gpu_residuals_1(NULL),
gpu_residuals_2(NULL)
{
	strncpy(this->name, name, 8);


	// *** Layers allocation *** //

	this->bn1_layer = new BatchNormFullPrecLayer(params_bn1.name,
												 params_bn1.in_width,
												 params_bn1.in_height,
												 params_bn1.in_channels,
												 params_bn1.scale,
												 params_bn1.shift);

	/*this->tr_layer = new TransposeFullPrecLayer("tr",
											    this->input_width,
											    this->input_height,
												this->input_channels);*/

	/*this->sum_layer = new MatrixSumLayer("sum",
										 this->input_width,
										 this->input_height * this->input_channels);*/
}

void MLPMixer::release()
{
	std::cout << "Dealloc instance / CUDA resources..." << std::endl;


	// Dealloc data space (may be NULL in case this layer is connected to other layers).
	if(this->input_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->input_gpu) );
		this->input_gpu = NULL;
	}

	if(this->output_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->output_gpu) );
		this->output_gpu = NULL;
	}


	// Dealloc objects allocated in the heap.
	if(this->bn1_layer != NULL)
		delete this->bn1_layer;

	if(this->bn2_layer != NULL)
		delete this->bn2_layer;

	if(this->bmm1_layer != NULL)
		delete this->bmm1_layer;

	if(this->bmm2_layer != NULL)
		delete this->bmm2_layer;

	if(this->bmm3_layer != NULL)
		delete this->bmm3_layer;

	if(this->bmm4_layer != NULL)
		delete this->bmm4_layer;

	if(this->tr_layer != NULL)
		delete this->tr_layer;

	if(this->sum_layer != NULL)
		delete this->sum_layer;
}

MLPMixer* MLPMixer::ready()
{
	// Various checks...
	if(this->input_gpu == NULL || this->input_size() == 0)
	{
		std::cout << "ERROR: Input data has not been allocated/initialized on the GPU." << std::endl;
		exit(1);
	}

	if(this->output_gpu == NULL || this->output_size() == 0)
	{
		std::cout << "ERROR: Output has not been allocated/initialized on the GPU." << std::endl;
		exit(1);
	}

	if(this->bn1_layer == NULL || this->bn2_layer == NULL)
	{
		std::cout << "ERROR: At least one of the batch normalization layers has not been initialized." << std::endl;
		exit(1);
	}

	if(this->bmm1_layer == NULL || this->bmm2_layer == NULL || this->bmm3_layer == NULL || this->bmm4_layer == NULL)
	{
		std::cout << "ERROR: At least one of the binary matrix multiplication layers has not been initialized." << std::endl;
		exit(1);
	}

	if(this->tr_layer == NULL || this->sum_layer == NULL)
	{
		std::cout << "ERROR: Either the matrix transposition or the sum layer has not been initialized." << std::endl;
		exit(1);
	}


	// Return the pointer to the shadow copy (to be used within a kernel).
	return this;
}

void MLPMixer::execute_layer()
{

}
