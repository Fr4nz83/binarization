#pragma once

#include "../generator.h"
#include "layer_MLPMixer.cuh"

BatchNormFullPrecLayer::BatchNormLayerParams generate_random_bn(const char* name,
													  	  	 	const unsigned& in_width,
																const unsigned& in_height,
																const unsigned& in_channels)
{
	float *scale = gen_matrix_ptr(1, in_channels);
	float *shift = gen_matrix_ptr(1, in_channels);

	return {name, in_width, in_height, in_channels, scale, shift};
}
