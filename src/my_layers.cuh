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


/**
 * @brief This class provides the interface that all the layers must implement.
 */
class Layer
{
public:

	// *** CTORS / DTOR *** //

	virtual ~Layer() {};



	// *** PUBLIC METHODS *** //

	virtual void release() = 0;
	virtual Layer* ready() = 0;

	virtual int get_size_batch() = 0;

	virtual int input_size() = 0;
	virtual int input_bytes() = 0;
	virtual int get_input_width() = 0;
	virtual int get_input_heigth() = 0;
	virtual int get_input_channels() = 0;

	virtual int output_size() = 0;
	virtual int output_bytes() = 0;
	virtual int get_output_width() = 0;
	virtual int get_output_height() = 0;
	virtual int get_output_channels() = 0;

	virtual void allocate_output_gpu() = 0;
	virtual void load_input_gpu(const unsigned& size_batch, const std::vector<void*>& input) = 0;
	virtual void set_input_gpu(const unsigned& size_batch, const std::vector<void*>& input_gpu) = 0;

	virtual void* get_output_gpu() = 0;
	virtual void download_output_gpu(void* output) = 0;

	virtual void execute_layer() = 0;
};


/** @brief This class implements a batch normalization layer which can operate on different channels.
 *
 *  @note The implementation assumes that a dataset of images follows the NCHW format.
 */
class BatchNormFullPrecLayer : public Layer
{
protected:

	// *** PROTECTED FIELDS *** //

	unsigned size_batch;

	float* input_gpu;

	unsigned input_width;
	unsigned input_height;
	unsigned input_channels;

	float* scale_gpu;
	float* shift_gpu;

	BatchNormFullPrecLayer* gpu; // GPU shadow.
	char name[8];



	// *** FRIEND FUNCTIONS *** //

	friend __global__ void BNFPLayer(BatchNormFullPrecLayer* p);



public:

	// *** PUBLIC CTORS/DTOR *** //

	BatchNormFullPrecLayer(const char* name,
						   const unsigned& in_width,
						   const unsigned& in_height,
						   const unsigned& in_channels,
						   const float* scale,
						   const float* shift);
	virtual ~BatchNormFullPrecLayer() {this->release();};



	// *** PUBLIC METHODS *** //

	inline virtual void release();
	inline virtual BatchNormFullPrecLayer* ready();

	inline virtual int get_size_batch() {return this->size_batch;}

	inline virtual int input_size() {return this->input_channels * this->input_height * this->input_width * this->size_batch;}
	inline virtual int input_bytes() {return this->input_size() * sizeof(float);}
	inline virtual int get_input_width() {return this->input_width;}
	inline virtual int get_input_heigth() {return this->input_height;}
	inline virtual int get_input_channels() {return this->input_channels;}

	inline virtual int output_size() {return this->input_size();}
	inline virtual int output_bytes() {return this->input_bytes();}
	inline virtual int get_output_width() {return this->input_width;}
	inline virtual int get_output_height() {return this->input_height;}
	inline virtual int get_output_channels() {return this->input_channels;}

	inline virtual void allocate_output_gpu() {};
	inline virtual void load_input_gpu(const unsigned& size_batch, const std::vector<void*>& input);
	inline virtual void set_input_gpu(const unsigned& size_batch, const std::vector<void*>& input_gpu)
	{
		this->input_gpu = static_cast<float*>(input_gpu[0]);
		this->size_batch = size_batch;
	}

	inline virtual void* get_output_gpu() {auto tmp = this->input_gpu; this->input_gpu = NULL; return tmp;}
	inline virtual void download_output_gpu(void* output);

	inline virtual void execute_layer();
};

// Pull in the definitions of the methods associated with the class BatchNormFullPrecLayer.
#include "my_layers_batch_norm.inl"



class TransposeFullPrecLayer : public Layer
{
protected:

	// *** PROTECTED FIELDS *** //

	unsigned size_batch;

	float* input_gpu;
	float* output_gpu;

	unsigned input_width;
	unsigned input_height;
	unsigned input_channels;

	TransposeFullPrecLayer* gpu; // GPU shadow.
	char name[8];



	// *** FRIEND FUNCTIONS *** //

	friend __global__ void TransposeFPLayer(TransposeFullPrecLayer* p);



public:

	// *** PUBLIC CTORS/DTOR *** //

	TransposeFullPrecLayer(const char* name,
						   const unsigned& data_width,
						   const unsigned& data_height,
						   const unsigned& data_channels);
	virtual ~TransposeFullPrecLayer(){this->release();};



	// *** PUBLIC METHODS *** //

	inline virtual void release();
	inline virtual TransposeFullPrecLayer* ready();

	inline virtual int get_size_batch() {return this->size_batch;}

	inline virtual int input_size() {return this->input_channels * this->input_height * this->input_width * this->size_batch;}
	inline virtual int input_bytes() {return this->input_size() * sizeof(float);}
	inline virtual int get_input_width() {return this->input_width;}
	inline virtual int get_input_heigth() {return this->input_height;}
	inline virtual int get_input_channels() {return this->input_channels;}

	inline virtual int output_size() {return this->input_size();}
	inline virtual int output_bytes() {return this->input_bytes();}
	inline virtual int get_output_width() {return this->input_height;}
	inline virtual int get_output_height() {return this->input_width;}
	inline virtual int get_output_channels() {return this->input_channels;}

	inline virtual void allocate_output_gpu();
	inline virtual void load_input_gpu(const unsigned& size_batch, const std::vector<void*>& input);
	inline virtual void set_input_gpu(const unsigned& size_batch, const std::vector<void*>& input_gpu)
	{
		this->input_gpu = static_cast<float*>(input_gpu[0]);
		this->size_batch = size_batch;
	}

	inline virtual void* get_output_gpu() {auto tmp = this->output_gpu; this->output_gpu = NULL; return tmp;}
	inline virtual void download_output_gpu(void* output);

	inline virtual void execute_layer();
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

	inline void execute_layer();
};

// Pull in the definitions of the methods associated with the class ConvLayer.
#include "my_layers_conv.inl"



class BinaryMultiplicationLayer : public Layer
{
protected:

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



	// *** FRIEND CUDA KERNELS *** //

	friend __global__ void PackWeight32(const float* __restrict__ A, unsigned* B,
								 	    const int A_height, const int A_width);
	friend __global__ void Input_Binarization(BinaryMultiplicationLayer *p);
	friend __global__ void Mat_BinMul(BinaryMultiplicationLayer* p);
	friend __global__ void Mat_BinMul_T(BinaryMultiplicationLayer* p);
	friend __global__ void Mat_BinMul_OutBin(BinaryMultiplicationLayer* p);
	friend __global__ void Mat_BinMul_T_OutBin(BinaryMultiplicationLayer* p);



	// *** PROTECTED METHODS *** //

	inline void init_bin_weights(const float* weights, const float* bias);



public:

	// *** PUBLIC CTORS/DTOR *** //

	BinaryMultiplicationLayer(const char* name,
						   	  const unsigned& weigths_height,
							  const unsigned& weigths_width,
							  const float* weights,
							  const float* bias,
							  const bool& binarized_input = false,
							  const bool& binarized_output = false,
							  const bool& transpose_output = false,
							  const bool& apply_gelu = true);
	virtual ~BinaryMultiplicationLayer(){this->release();};



	// *** PUBLIC METHODS *** //

	inline virtual void release();
	inline virtual BinaryMultiplicationLayer* ready();

	inline virtual int get_size_batch() {return this->get_input_heigth();} // NOTE: this output of this function is effectively meaningless if
																		   // we're just performing a simple matrix multiplication.

	inline virtual int input_size() {return this->input_height * this->input_width;}
	inline virtual int input_bytes() {return this->input_size() * sizeof(float);}
    inline int input_bit_size() {return FEIL(this->input_height) * CEIL(this->input_width);}
    inline int input_bit_bytes() {return input_bit_size() * sizeof(unsigned);}
	inline virtual int get_input_width() {return this->input_width;}
	inline virtual int get_input_heigth() {return this->input_height;}
	inline virtual int get_input_channels() {return 1;} // NOTE: the output of this function is meaningless in this layer.

	// The function below returns the pointer to the binarized input.
	// NOTE: used only for debugging purposes.
	inline unsigned* get_ptr_input_bin_gpu() {auto tmp = this->input_bin_gpu; this->input_bin_gpu = NULL; return tmp;}

	inline virtual int output_size() {return this->input_height * this->weights_width;}
	inline virtual int output_bytes() {return this->output_size() * sizeof(float);}
    inline int output_bit_size() {return !this->transpose_output ? FEIL(this->input_height) * CEIL(this->weights_width) :
    															   CEIL(this->input_height) * FEIL(this->weights_width);}
    inline int output_bit_bytes() {return output_bit_size() * sizeof(unsigned);}
	inline int virtual get_output_width() {return !this->transpose_output ? this->weights_width : this->input_height;}
	inline int virtual get_output_height() {return !this->transpose_output ? this->input_height : this->weights_width;}
	inline virtual int get_output_channels() {return get_input_channels();} // NOTE: the output of this function is meaningless in this layer.

	inline int weights_size() {return this->weights_width * this->weights_height;}
	inline int weight_bytes() {return this->weights_size() * sizeof(float);}
	inline int weight_bit_size() {return CEIL(this->weights_height) * FEIL(this->weights_width);}
	inline int weight_bit_bytes() {return weight_bit_size() * sizeof(unsigned);}
	inline int get_weights_width() {return this->weights_width;}
	inline int get_weights_height() {return this->weights_height;}

	inline virtual void allocate_output_gpu();
	inline virtual void load_input_gpu(const unsigned& size_batch, const std::vector<void*>& input);
	inline virtual void set_input_gpu(const unsigned& size_batch, const std::vector<void*>& input);

	inline virtual void* get_output_gpu()
	{
		// NOTE: the set of assignments below transfer the responsibility of the management of
		// the GPU output pointer to the caller (hence the NULL).
		void* tmp = !this->binarize_output ? (void*)this->output_gpu : (void*)this->output_bin_gpu;
		if(this->binarize_output == false) this->output_gpu = NULL;
		else this->output_bin_gpu = NULL;
		return tmp;
	}
	inline virtual void download_output_gpu(void* output);

	inline virtual void execute_layer();
};

// Pull in the definitions of the methods associated with the class ConvLayer.
#include "my_layers_binmult.inl"



class MatrixSumLayer : public Layer
{
protected:

	// *** PROTECTED FIELDS *** //

	float* input1_gpu;
	float* input2_gpu;
	float* output_gpu;

	unsigned input_width;
	unsigned input_height;

	MatrixSumLayer* gpu; // GPU shadow.
	char name[8];



	// *** CUDA KERNEL FRIENDS *** //

	friend __global__ void Matrix_Sum(MatrixSumLayer* p);



public:

	// *** PUBLIC CTORS/DTOR *** //

	MatrixSumLayer(const char* name,
			  	   const unsigned& data_width,
				   const unsigned& data_height);
	virtual ~MatrixSumLayer(){this->release();};



	// *** PUBLIC METHODS *** //

	inline virtual void release();
	virtual MatrixSumLayer* ready();

	inline virtual int get_size_batch() {return 1;}

	inline virtual int input_size() {return this->input_height * this->input_width;}
	inline virtual int input_bytes() {return this->input_size() * sizeof(float);}
	inline virtual int get_input_width() {return this->input_width;}
	inline virtual int get_input_heigth() {return this->input_height;}
	inline virtual int get_input_channels() {return 1;}

	inline virtual int output_size() {return this->input_size();}
	inline virtual int output_bytes() {return this->input_bytes();}
	inline virtual int get_output_width() {return this->get_input_width();}
	inline virtual int get_output_height() {return this->get_input_heigth();}
	inline virtual int get_output_channels() {return this->get_input_channels();}

	inline virtual void allocate_output_gpu();
	inline virtual void load_input_gpu(const unsigned& size_batch, const std::vector<void*>& input);
	inline virtual void set_input_gpu(const unsigned& size_batch, const std::vector<void*>& input_gpu)
	{
		this->input1_gpu = static_cast<float*>(input_gpu[0]);
		this->input2_gpu = static_cast<float*>(input_gpu[1]);
	}

	inline virtual void* get_output_gpu() {auto tmp = this->output_gpu; this->output_gpu = NULL; return tmp;}
	inline virtual void download_output_gpu(void* output);

	inline virtual void execute_layer();
};

// Pull in the definitions of the methods associated with the class ConvLayer.
#include "my_layers_matrix_sum.inl"
