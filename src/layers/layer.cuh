/*
 * my_layers.cuh
 *
 *  Created on: 9 giu 2022
 *      Author: francesco lettich
 */

#pragma once


// *** INCLUDES *** //
#include "../cuda_utilities.cuh"


/**
 * @brief This class provides the interface that all the layers must implement.
 */
class Layer
{
public:

	// *** PUBLIC CTORS / DTOR *** //

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
