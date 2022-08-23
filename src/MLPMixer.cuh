/*
 * MLPMixer.h
 *
 *  Created on: 1 ago 2022
 *      Author: lettich
 */
#pragma once


/*** INCLUDES ***/

#include "my_layers.cuh"



class MLPMixer
{
public:

	// *** FIELDS *** //

	// Input fields.
	float* input_gpu;			// Memory allocated for FP input.
	unsigned input_height;
	unsigned input_width;
	unsigned input_channels;
	unsigned input_size_batch;

	// Output fields.
	float* output_gpu;			// Memory allocated for FP output.


	// Residuals that at some points are used within the MLP-Mixer block.
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



	// *** CTORS / DTOR *** //

	MLPMixer(const char* name,
			 BatchNormFullPrecLayer* bn1_layer,
	         BinaryMultiplicationLayer* bmm1_layer,
	         BinaryMultiplicationLayer* bmm2_layer,
	         BatchNormFullPrecLayer* bn2_layer,
	         BinaryMultiplicationLayer* bmm3_layer,
	         BinaryMultiplicationLayer* bmm4_layer);
	~MLPMixer(){this->release();};



	// *** METHODS *** //

	inline void release() {};

	inline int input_size() {return this->input_size_batch * this->input_height * this->input_width * this->input_channels;}
	inline int input_bytes() {return this->input_size() * sizeof(float);}
	inline int get_input_width() {return this->input_width;}
	inline int get_input_heigth() {return this->input_height;}
	inline int get_input_channels() {return this->input_channels;}
	inline int get_input_size_batch() {return this->input_size_batch;}

	inline int output_size() {return this->input_size();}
	inline int output_bytes() {return this->input_bytes();}
	inline int get_output_width() {return this->get_input_width();}
	inline int get_output_height() {return this->get_input_heigth();}
	inline int get_output_channels() {return this->get_input_channels();};
	inline int get_output_size_batch() {return this->get_input_size_batch();};

	// TODO: da implementare.
	inline void allocate_output_gpu() {};
	inline void load_input_gpu(void* input, const unsigned& input_height, const unsigned& input_width, const unsigned& input_channels) {};
	inline void set_input_gpu(void* input_gpu, const unsigned& input_height, const unsigned& input_width, const unsigned& input_channels) {};

	// TODO: da implementare.
	inline void* get_output_gpu() {return 0;};
	inline void download_output_gpu(void* output) {};

	// TODO: da implementare.
	inline void execute_layer() {};
};

#include "MLPMixer.inl"
