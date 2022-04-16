#pragma once


#include "utils.h"
#include <vector>
#include <cusparse.h>
#include <cusparseLt.h>
#include "cublas_v2.h"

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"


class GPU_MatMul
{
	protected :

	/*** PROTECTED FIELDS ***/

	cublasHandle_t handleBlas;

	// Data structures associated with the dense layers in the DNN.
	const std::vector<DenseMatrix>& vec_dense;
	std::vector<thrust::device_vector<float>> d_weights_dense;

	// Data structures associated with the input dataset.
	float *d_input;

	// Data structures associated with GPU output management.
	thrust::device_vector<float> d_output;
	
	// Data structures used to measure time elapsed on GPU.
	float time_load, time_compute, time_output;


	/*** PROTECTED CTORS ***/

	// GPU_MatMul() {};



	/*** PROTECTED METHODS ***/

	void load_dense_layers_GPU();


	void load_dataset_GPU(void* src,
			      void** dest,
			      const uint32_t& numFeats,
			      const uint32_t& batch_size,
			      const uint32_t& size_type);


	void compute_DD(const int& batch_size,
			const uint32_t& idx_layer);



	public:

	/*** PUBLIC CTORS ***/

	GPU_MatMul(const std::vector<DenseMatrix>& vec_dense);
	~GPU_MatMul();



	/*** PUBLIC METHODS ***/
	
	float get_time_load() {return time_load;};
	float get_time_compute() {return time_compute;};
	float get_time_output() {return time_output;};

	void* score_GPU(const int batch_size,
			float* input,
			const uint32_t numFeats);
};
