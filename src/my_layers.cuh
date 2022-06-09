/*
 * my_layers.cuh
 *
 *  Created on: 9 giu 2022
 *      Author: francesco lettich
 */

#pragma once


#include "sbnn32_param.h"

class BatchNormLayer
{
public:

	//Input
	unsigned* input;
	unsigned* input_gpu;
	unsigned input_width;
	unsigned input_height;
	unsigned input_channels;

	//Output
	unsigned* output;
	unsigned* output_gpu;
	unsigned output_width;
	unsigned output_height;
	unsigned output_channels;
};


class ConvLayer
{
public:

	// *** FIELDS *** //

	//Input
	unsigned* input;
	unsigned* input_gpu;
	unsigned input_width;
	unsigned input_height;
	unsigned input_channels;

	//Weight
	float* filter;
	unsigned* filter_gpu;
	unsigned filter_width;
	unsigned filter_height;

	//Output
	unsigned* output;
	unsigned* output_gpu;
	unsigned output_width;
	unsigned output_height;
	unsigned output_channels;

	// Batch normalization
	// float* bn;
	// float* bn_gpu;

	// Other fields.
	unsigned batch;
	unsigned stride_vertical;
	unsigned stride_horizontal;
	unsigned pad_h;
	unsigned pad_w;

	//GPU shadow
	// ConvLayer* gpu;

	char name[8];


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
			  bool same_padding = true)
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


		// Calculate the padding.
		this->pad_h = same_padding?((( (input_height+stride_vertical-(input_height%stride_vertical))
						/stride_vertical-1)*stride_vertical+filter_height-input_height)>>1):0;
		this->pad_w = same_padding?((( (input_width+stride_horizontal-(input_width%stride_horizontal))
							/stride_horizontal-1)*stride_horizontal+filter_width-input_width)>>1):0;


		auto pool_height = 0;
		auto pool_height = 0;
		auto buf_height = 0;
		auto buf_width = 0;

		if (pool_height == 0)
		{
			output_height = same_padding?(input_height+stride_vertical-1)/stride_vertical
				:((input_height-filter_height)/stride_vertical+1);
			this->buf_height = 0;
		}
		else
		{
			buf_height = same_padding?(input_height+stride_vertical-1)/stride_vertical
				:((input_height-filter_height)/stride_vertical+1);
			output_height = (buf_height+pool_height-1)/pool_height;//pooling height
		}


		if (pool_width == 0)
		{
			output_width = same_padding?(input_width+stride_horizontal-1)/stride_horizontal
				:((input_width-filter_width)/stride_horizontal+1);
			this->buf_width = 0;
		}
		else
		{
			buf_width = same_padding?(input_width+stride_horizontal-1)/stride_horizontal
				:((input_width-filter_width)/stride_horizontal+1);
			output_width = (buf_width+pool_width-1)/pool_width; //pooling width
		}
	}

	~ConvLayer() { release(); }


	ConvLayer* ready()
	{
		if (input_gpu == NULL)
		{
			fprintf(stderr, "Input data has not been uploaded to GPU.\n");
			exit(1);
		}
		if (output_gpu == NULL)
		{
			fprintf(stderr, "Output on GPU has not been allocated.\n");
			exit(1);
		}
		if (save_residual && save_residual_gpu == NULL)
		{
			fprintf(stderr, "Residual for saving on GPU has not been allocated.\n");
			exit(1);
		}
		if (inject_residual && inject_residual_gpu == NULL)
		{
			/*fprintf(stderr, this->name);*/

			fprintf(stderr, "Residual for injecting on GPU has not been allocated.\n");
			exit(1);
		}
		CUDA_SAFE_CALL( cudaMalloc((void**)&(this->gpu), sizeof(Conv32LayerParam)) );
		CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this,
					sizeof(Conv32LayerParam), cudaMemcpyHostToDevice) );
		return this->gpu;
	}

	void set_input_gpu(unsigned* input_gpu)
	{
		this->input_gpu = input_gpu;
	}

	Conv32LayerParam* initialize(FILE* config_file, unsigned* prev_layer_gpu, int* inject_residual_gpu = NULL)
	{
		//Process weight
		this->filter = (float*)malloc(filter_bytes());
		launch_array(config_file, this->filter, filter_size());
		CUDA_SAFE_CALL( cudaMalloc((void**)&(this->filter_gpu), filter_bit_bytes()) );


		float* filter_float = NULL;
		CUDA_SAFE_CALL( cudaMalloc((void**)&(filter_float), filter_bytes()) );
		CUDA_SAFE_CALL( cudaMemcpy(filter_float, filter,
					filter_bytes(), cudaMemcpyHostToDevice) );

		//Binarize Filter
		PackFiltersByInChannels32<<<dim3(filter_height*filter_width, output_channels), 32>>>(
			filter_float, filter_gpu, input_channels, output_channels,
			filter_width, filter_height);
		CUDA_SAFE_CALL( cudaFree(filter_float) );

		//Process bn
		this->bn = (float*)malloc(bn_bytes());
		launch_array(config_file, this->bn, bn_size());
		CUDA_SAFE_CALL( cudaMalloc((void**)&(this->bn_gpu), bn_bytes()) );
		CUDA_SAFE_CALL( cudaMemcpy(bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice) );

		//Allocate output gpu
		CUDA_SAFE_CALL( cudaMalloc((void**)&(this->output_gpu), output_bit_bytes()) );
		CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );

		set_input_gpu(prev_layer_gpu);

		//Allocate residual for saving
		if (save_residual)
		{
			CUDA_SAFE_CALL( cudaMalloc((void**)&(this->save_residual_gpu), output_bytes()) );
			CUDA_SAFE_CALL( cudaMemset(this->save_residual_gpu, 0, output_bytes()) );
		}

		//inject residual
		if (inject_residual) set_inject_residual_gpu(inject_residual_gpu);

		return this->ready();
	}

	int input_size() { return  input_channels*input_height*input_width*batch;}
	int input_bytes() { return input_size()*sizeof(unsigned);}
	int input_bit_size() { return  CEIL(input_channels)*input_height*input_width*batch;}
	int input_bit_bytes() { return input_bit_size()*sizeof(unsigned);}

	int filter_size() { return output_channels*input_channels*filter_height*filter_width;}
	int filter_bytes() { return filter_size()*sizeof(float);}
	int filter_bit_size() {return output_channels*FEIL(input_channels)*filter_height*filter_width;}
	int filter_bit_bytes() { return output_channels*CEIL(input_channels)
		*filter_height*filter_width*sizeof(unsigned);}

	int output_size() { return output_channels*output_height*output_width*batch;}
	int output_bytes() { return output_size()*sizeof(unsigned);}
	int output_bit_size()
	{
		return output_transpose ?
			   FEIL(output_channels)*output_height*output_width*FEIL(batch) :
			   FEIL(output_channels)*output_height*output_width*batch;
	}
	int output_bit_bytes()
	{
		return output_transpose?CEIL(output_channels)*output_height*output_width*
			FEIL(batch)*sizeof(unsigned): CEIL(output_channels)*output_height*
			output_width*batch*sizeof(unsigned);
	}

	int bn_size() { return output_channels;}
	int bn_bytes() { return bn_size()*sizeof(float);}

	unsigned* get_output_gpu()
	{
		return this->output_gpu;
	}
	int* get_residual_gpu()
	{
		return this->save_residual_gpu;
	}

	void release()
	{
		if (this->filter!=NULL) {free(this->filter); this->filter=NULL;}
		if (this->bn!=NULL) {free(this->bn); this->bn=NULL;}
		if (this->output!=NULL) {free(this->output); this->output=NULL;}
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
};
