/*
 * my_layers.cuh
 *
 *  Created on: 9 giu 2022
 *      Author: francesco lettich
 */

#pragma once


// *** INCLUDES *** //
#include <cudnn.h>
#include "cuda_utilities.cuh"



/** @brief This class implements a batch normalization layer which can operate on different channels.
 *
 *  @note The implementation assumes that a dataset of images follows the NCHW format.
 */
class BatchNormFullPrecLayer
{
public:

	// *** FIELDS *** //
	unsigned size_batch;

	float* input_gpu;

	unsigned input_width;
	unsigned input_height;
	unsigned input_channels;

	float* scale_gpu;
	float* shift_gpu;

	// GPU shadow.
	BatchNormFullPrecLayer* gpu;

	char name[8];



	// *** CTORS/DTOR *** //

	BatchNormFullPrecLayer(const char* name,
						   const unsigned& in_width,
						   const unsigned& in_height,
						   const unsigned& in_channels,
						   const float* scale,
						   const float* shift);
	~BatchNormFullPrecLayer(){this->release();};



	// *** METHODS *** //

	void release();

	BatchNormFullPrecLayer* ready();

	inline int get_size_batch() {return this->size_batch;}
	inline int input_size() {return this->input_channels * this->input_height * this->input_width * this->size_batch;}
	inline int input_bytes() {return this->input_size() * sizeof(float);}
	inline int get_input_width() {return this->input_width;}
	inline int get_input_heigth() {return this->input_height;}
	inline int get_input_channels() {return this->input_channels;}

	inline int output_size() {return this->input_size();}
	inline int output_bytes() {return this->input_bytes();}
	inline int get_output_width() {return this->input_width;}
	inline int get_output_height() {return this->input_height;}
	inline int get_output_channels() {return this->input_channels;}

	inline void allocate_output_gpu() {};
	inline void load_input_gpu(float* input, unsigned size_batch);
	inline void set_input_gpu(float* input_gpu, unsigned size_batch) { this->input_gpu = input_gpu; this->size_batch = size_batch;}

	inline float* get_output_gpu() {auto tmp = this->input_gpu; this->input_gpu = NULL; return tmp;}
	inline void download_output_gpu(float* output);
};

// Pull in the definitions of the methods associated with the class BatchNormFullPrecLayer.
#include "my_layers_batch_norm.inl"



class TransposeFullPrecLayer
{
public:

	// *** FIELDS *** //
	unsigned size_batch;

	float* input_gpu;
	float* output_gpu;

	unsigned input_width;
	unsigned input_height;
	unsigned input_channels;

	// GPU shadow.
	TransposeFullPrecLayer* gpu;

	char name[8];



	// *** CTORS/DTOR *** //

	TransposeFullPrecLayer(const char* name,
						   const unsigned& data_width,
						   const unsigned& data_height,
						   const unsigned& data_channels);
	~TransposeFullPrecLayer(){this->release();};



	// *** METHODS *** //

	inline void release();

	TransposeFullPrecLayer* ready();

	inline int input_size() {return this->input_channels * this->input_height * this->input_width * this->size_batch;}
	inline int input_bytes() {return this->input_size() * sizeof(float);}
	inline int get_input_width() {return this->input_width;}
	inline int get_input_heigth() {return this->input_height;}
	inline int get_input_channels() {return this->input_channels;}

	inline int output_size() {return this->input_size();}
	inline int output_bytes() {return this->input_bytes();}
	inline int get_output_width() {return this->input_height;}
	inline int get_output_height() {return this->input_width;}
	inline int get_output_channels() {return this->input_channels;}
	inline int get_size_batch() {return this->size_batch;}

	inline void allocate_output_gpu();
	inline void load_input_gpu(float* input, unsigned size_batch);
	inline void set_input_gpu(float* input_gpu, unsigned size_batch) {this->input_gpu = input_gpu; this->size_batch = size_batch;}

	inline float* get_output_gpu() {auto tmp = this->output_gpu; this->output_gpu = NULL; return tmp;}
	inline void download_output_gpu(float* output);
};

// Pull in the definitions of the methods associated with the class ConvLayer.
#include "my_layers_transpose.inl"



/** @brief This class implements a full-precision convolutional layer based on cuDNN.
 *
 *  @note The implementation assumes that a dataset of images follows the NCHW format, while the set filters follows the HWIO format.
 */
class ConvLayer
{
public:

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

	// Convolution general properties.
	bool same_conv;
	unsigned size_batch;
	unsigned stride_vertical;
	unsigned stride_horizontal;
	unsigned pad_h;
	unsigned pad_w;

	// Skip connections.
	bool save_residual;
	float *save_residual_gpu;

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
			  bool save_residual = true);
	~ConvLayer() {release();}



	// *** PUBLIC METHODS *** //

	bool ready();
	void release();

	bool initialize_filters(const float* filters);

	// I/O charachteristics.
	inline int input_size() {return this->input_channels*this->input_height*this->input_width*this->size_batch;}
	inline int input_bytes() {return input_size() * sizeof(float);}
	inline int get_input_size_batch() {return this->size_batch;}
	inline int get_input_width() {return this->input_width;}
	inline int get_input_heigth() {return this->input_height;}
	inline int get_input_channels() {return this->input_channels;}

	inline int output_size() {return this->output_channels * this->output_height * this->output_width * this->size_batch;}
	inline int output_bytes() {return this->output_size() * sizeof(float);}
	inline int get_output_width() {return this->output_width;}
	inline int get_output_height() {return this->output_height;}
	inline int get_output_channels() {return this->output_channels;}

	// Filters charachteristics.
	inline int filter_size() {return this->output_channels * this->input_channels * this->filter_height * this->filter_width;}
	inline int filter_bytes() {return this->filter_size() * sizeof(float);}

	inline bool allocate_output_gpu();
	inline bool load_input(const unsigned& batch_size, const float* img_data);
	inline bool set_input_gpu(float* input_gpu, const unsigned& batch_size);

	inline float* get_output_gpu() {auto tmp = this->output_gpu; this->output_gpu = NULL; return tmp;}
	void download_output_gpu(float* output);

	inline float* get_residual_gpu(){auto tmp = this->save_residual_gpu; this->save_residual_gpu = NULL; return tmp;}
	void download_residual_gpu(float* residual);

	bool execute_layer();
};

// Pull in the definitions of the methods associated with the class ConvLayer.
#include "my_layers_conv.inl"



class BinaryMultiplicationLayer
{
public:

	// *** PROTECTED FIELDS *** //

	// Input fields.
	float* input_gpu;			// Memory allocated for FP input.
	unsigned* input_bin_gpu;	// Memory allocated for binarized input.
	unsigned input_height;
	unsigned input_width;

	// Weights fields.
	unsigned* weights_gpu;
	unsigned weights_height; // Number of features in the input.
	unsigned weights_width; // Number of activation units.

	// Bias.
	float* bias_gpu;

	// Output fields.
	float* output_gpu;
	unsigned* output_bin_gpu;

	// GPU shadow.
	BinaryMultiplicationLayer* gpu;

	// Layer general properties.
	char name[8];
	bool binarized_input;
	bool binarize_output;
	bool transpose_output;
	bool apply_gelu;



	// *** CTORS/DTOR *** //

	BinaryMultiplicationLayer(const char* name,
						   	  const unsigned& weigths_height,
							  const unsigned& weigths_width,
							  const float* weights,
							  const float* bias,
							  const bool& binarized_input = false,
							  const bool& binarized_output = false,
							  const bool& transpose_output = false,
							  const bool& apply_gelu = true);
	~BinaryMultiplicationLayer(){this->release();};



	// *** METHODS *** //

	inline void release();

	inline void init_bin_weights(const float* weights, const float* bias);
	BinaryMultiplicationLayer* ready();

	inline int input_size() {return this->input_height * this->input_width;}
	inline int input_bytes() {return this->input_size() * sizeof(float);}
    int input_bit_size() {return FEIL(this->input_height) * CEIL(this->input_width);}
    int input_bit_bytes() {return input_bit_size() * sizeof(unsigned);}
	inline int get_input_width() {return this->input_width;}
	inline int get_input_heigth() {return this->input_height;}

	// The function below returns the pointer to the binarized input.
	// NOTE: used only for debugging purposes.
	inline unsigned* get_ptr_input_bin_gpu() {auto tmp = this->input_bin_gpu; this->input_bin_gpu = NULL; return tmp;}

	inline int output_size() {return this->input_height * this->weights_width;}
	inline int output_bytes() {return this->output_size() * sizeof(float);}
    int output_bit_size() {return !this->transpose_output ? FEIL(this->input_height) * CEIL(this->weights_width) :
    														CEIL(this->input_height) * FEIL(this->weights_width);}
    int output_bit_bytes() {return output_bit_size() * sizeof(unsigned);}
	inline int get_output_width() {return !this->transpose_output ? this->weights_width : this->input_height;}
	inline int get_output_height() {return !this->transpose_output ? this->input_height : this->weights_width;}

	inline int weights_size() {return this->weights_width * this->weights_height;}
	inline int weight_bytes() {return this->weights_size() * sizeof(float);}
	int weight_bit_size() {return CEIL(this->weights_height) * FEIL(this->weights_width);}
	int weight_bit_bytes() {return weight_bit_size() * sizeof(unsigned);}
	inline int get_weights_width() {return this->weights_width;}
	inline int get_weights_height() {return this->weights_height;}

	inline void allocate_output_gpu();
	inline void load_input_gpu(void* input, unsigned input_height);
	inline void set_input_gpu(void* input_gpu, unsigned input_height);

	inline void* get_output_gpu()
	{
		// NOTE: the set of assignments below transfer the responsibility of the management of
		// the GPU output pointer to the caller (hence the NULL).
		void* tmp = !this->binarize_output ? (void*)this->output_gpu : (void*)this->output_bin_gpu;
		if(this->binarize_output == false) this->output_gpu = NULL;
		else this->output_bin_gpu = NULL;
		return tmp;
	}
	inline void download_output_gpu(void* output);

	inline void execute_layer();
};

// Pull in the definitions of the methods associated with the class ConvLayer.
#include "my_layers_binmult.inl"



class MatrixSumLayer
{
public:

	// *** FIELDS *** //

	float* input1_gpu;
	float* input2_gpu;
	float* output_gpu;

	unsigned input_width;
	unsigned input_height;

	// GPU shadow.
	MatrixSumLayer* gpu;

	char name[8];



	// *** CTORS/DTOR *** //

	MatrixSumLayer(const char* name,
			  const unsigned& data_width,
			  const unsigned& data_height);
	~MatrixSumLayer(){this->release();};



	// *** METHODS *** //

	inline void release();

	MatrixSumLayer* ready();

	inline int input_size() {return this->input_height * this->input_width;}
	inline int input_bytes() {return this->input_size() * sizeof(float);}
	inline int get_input_width() {return this->input_width;}
	inline int get_input_heigth() {return this->input_height;}

	inline int output_size() {return this->input_size();}
	inline int output_bytes() {return this->input_bytes();}
	inline int get_output_width() {return this->get_input_width();}
	inline int get_output_height() {return this->get_input_heigth();}

	inline void allocate_output_gpu();
	inline void load_input_gpu(float* input1, float* input2);
	inline void set_input_gpu(float* input1_gpu, float* input2_gpu) {this->input1_gpu = input1_gpu; this->input2_gpu = input2_gpu;}

	inline float* get_output_gpu() {auto tmp = this->output_gpu; this->output_gpu = NULL; return tmp;}
	inline void download_output_gpu(float* output);

	inline void execute_layer();
};

// Pull in the definitions of the methods associated with the class ConvLayer.
#include "my_layers_matrix_sum.inl"
