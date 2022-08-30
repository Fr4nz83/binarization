/*
 * gpu_mat.cpp
 *
 *  Created on: 10 ott 2021
 *      Author: lettich
 */


// *** INCLUDES *** //
#include <iostream>
#include "cuda_runtime.h"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cstdio>             // printf
#include <cstdlib>            // std::rand

#include "gpu_mat.h"


// *** MACROS *** //
// #define _CUDA_PROFILING_
#ifdef _CUDA_PROFILING_
#include "cuda_profiler_api.h"
#endif



// *** HELPER FUNCTORS SPECIFIC TO THIS OBJECT *** //

/*
 * @brief Template operatore unario binarizzazione.
 */
template <typename T>
struct binarize
{
    __device__ T operator()(T& x)
    {
    	return x < 0. ? -1. : 1.;
    }
};



/*** PROTECTED METHODS ***/

void GPU_MatMul::load_dense_layers_GPU()
{
	std::cout << "Caricamento di tutti i layer in formato denso su GPU..." << std::endl;
	for(auto d : this->vec_dense)
	{
		std::cout << "Caricamento layer denso con dimensioni " << d.height << " x " << d.width << std::endl;

		// Save the weights in the original single precision float format.
		thrust::device_vector<float> tmp(d.weights);
		
		// Binarize the weights.
		thrust::transform(tmp.begin(),
				  tmp.end(),
				  tmp.begin(),
				  binarize<float>());
		
		this->d_weights_dense.push_back(thrust::device_vector<float>(tmp));
	}
	std::cout << "Layer caricati: " << this->d_weights_dense.size() << std::endl;
}

void GPU_MatMul::load_dataset_GPU(void* src,
				  void** dest,
				  const uint32_t& numFeats,
				  const uint32_t& batch_size,
				  const uint32_t& size_type)
{
	// *** Allocate on GPU the memory resources associated with the dense matrix B (i.e., the dataset). *** //
	// TODO: il caricamento del dataset e' il collo di bottiglia primario.
	//		 Possibili soluzioni: caricamento in batch del dataset con memcpy asincrone + pinned memory
	uint32_t dataset_size_bytes = numFeats * batch_size * size_type;


	// Classica copia host => GPU (si assume uso di pinned memory lato host per massimizzare il throughput).
#ifdef _CUDA_PROFILING_
	  cudaProfilerStart();
#endif
	cudaMalloc((void**)dest, dataset_size_bytes);
	cudaMemcpy(*dest, src, dataset_size_bytes, cudaMemcpyHostToDevice);
#ifdef _CUDA_PROFILING_
	  cudaProfilerStop();
#endif

	// Use of the unified memory. Assumes that cudaHostRegister has already been applied on "input".
	// NOTE: Seems very slow!
	// this->d_input = input;

	std::cout << "Size of the dataset: " << dataset_size_bytes << std::endl;
}

void GPU_MatMul::compute_DD(const int& batch_size, const uint32_t& idx_layer)
{
	cublasStatus_t status;


	const auto& info_layer = this->vec_dense[idx_layer];
	const auto& d_weights = this->d_weights_dense[idx_layer];
	uint32_t num_units = info_layer.height;
	uint32_t num_inputs = info_layer.width;


	// 1 - "Swappiamo (via move) il contenuto di d_output con quello di d_in_layer. Quindi, facciamo il resize di d_output
	// 	   in maniera tale che contenga il contenuto di questo
	thrust::device_vector<float> d_in_layer; d_in_layer.swap(this->d_output);


	std::cout << "Dimensione matrice pesi layer: " << num_units << " x " << num_inputs << std::endl;
	this->d_output.resize(batch_size * num_units);


	// ATTENZIONE: va considerato che cuBLAS considera le matrici come column-major!
	// If you want the matrix C to be stored in row-major, you cam calculate the C^T stored in column-major with the formula C^T = B^T * A^T.
	// B and A are already transposed if they are passed in row major format, thus it becomes a matter of specifying the correct values
	// to the params m, n, k, lda, ldb, and ldc.
	float alpha = 1.0;
	float beta = 0.0;
#ifdef _CUDA_PROFILING_
	  cudaProfilerStart();
#endif
	status = cublasSgemm(this->handleBlas,
						 CUBLAS_OP_N,
						 CUBLAS_OP_N,
						 batch_size, // m
						 num_units,  // n
						 num_inputs, // k -- NOTA: il numero di colonne e' in verita' il numero di righe della matrice quando questa e' in row major format.
						 &alpha,
						 idx_layer ? thrust::raw_pointer_cast(d_in_layer.data()) : this->d_input, // Qui passiamo in prima posizione l'input del layer precedente (o il dataset), che essendo in formato row major cublas considerera' la sua trasposta.
						 batch_size,	// leading size prima matrice.
						 thrust::raw_pointer_cast(d_weights.data()),  // Qui passiamo in seconda posizione la matrice dei pesi. Essendo in row major, cublas la considerera' di nuovo come trasposta.
						 num_inputs,    // leading size seconda matrice.
						 &beta,
						 thrust::raw_pointer_cast(this->d_output.data()),
						 batch_size);	// leading size matrice di output.
	if(CUBLAS_STATUS_SUCCESS == status) std::cout << "Dense matrix multi OK!" << std::endl;
#ifdef _CUDA_PROFILING_
	  cudaProfilerStop();
#endif


	// Binarize the output.
	// TODO: qui andra' scritto un piccolo kernel che opera come SBMM.
	thrust::transform(this->d_output.begin(),
			  this->d_output.end(),
			  this->d_output.begin(),
			  binarize<float>());


	// DEBUG
	/*std::cout << "Stampa risultato moltiplicazione..." << std::endl;
	thrust::host_vector<float> test = d_output;
	for(uint32_t row = 0; row < num_units; row++)
	{
		std::cout << "Stampa linea " << row << ":";
		for(uint32_t i = 0; i < batch_size; i++)
			std::cout << test[row * batch_size + i] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;*/


	std::cout << "Matrice di output di questo layer: " << batch_size << " x " << num_units << std::endl;
}


// *** PUBLIC CLASS CTORS *** //

/**
 * @brief: Class constructor.
 * @params s [in] This parameter represents the data structure containing the
 */
GPU_MatMul::GPU_MatMul(const std::vector<DenseMatrix>& vec_dense) :
vec_dense(vec_dense)
{
	// Memory check;
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	std::cout << "GPU INFO: Mem free at START: " << free << " (total: " << total << ")\n";

	// Initialize cuBLAS
	cublasStatus_t statusBlas;
	statusBlas = cublasCreate(&this->handleBlas);
	if(CUBLAS_STATUS_SUCCESS == statusBlas) std::cout << "Init handle cuBLAS OK!" << std::endl;

	// Load the data associated with the DNN.
	this->load_dense_layers_GPU(); // Load layers in dense format.
}



// *** DTOR *** //

GPU_MatMul::~GPU_MatMul()
{
	// Libera le risorse occupate dal contesto CUSPARSE.
	cublasDestroy(this->handleBlas);
}



/*** PUBLIC METHODS ***/

void* GPU_MatMul::score_GPU(const int batch_size,
			    float* input,
			    const uint32_t numFeats)
{
	std::cout << "*** Executing path performing sparse-dense & dense-dense multiplications with single precision floats! ***" << std::endl;
	
	
	// Data structures used to measure the execution time on the GPU.
	cudaEvent_t load_input, compute, output, end;
	cudaEventCreate(&load_input);
    	cudaEventCreate(&compute);
    	cudaEventCreate(&output);
    	cudaEventCreate(&end);
    	time_load = 0;
        time_compute = 0;
        time_output = 0;


	std::vector<double> total_time_layers(this->vec_dense.size(), 0);


	// Carica il dataset in memoria GPU.
	cudaEventRecord(load_input);
	void **dest = (void**)&this->d_input;
	this->load_dataset_GPU(static_cast<void*>(input), dest, numFeats, batch_size, sizeof(input[0]));
	
	
	// Binarize the input.
	cudaEventRecord(compute);
	thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(this->d_input);
	thrust::transform(dev_ptr, dev_ptr + (batch_size * numFeats), dev_ptr, binarize<float>());


	// *** Processa i layer densi *** //
	for(uint32_t i = 0; i < this->vec_dense.size(); i++)
	{
		std::cout << "Processing dense layer " << i << std::endl;

		this->compute_DD(batch_size, i);
		if(i == 0) cudaFree(this->d_input); // Free the device memory occupied by the dataset.
	}
	
	
	cudaEventRecord(output);
	float* out = new float[this->d_output.size()];
	cudaMemcpy(out, thrust::raw_pointer_cast(this->d_output.data()), this->d_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	    
	    
	cudaEventElapsedTime(&time_load, load_input, compute);
    	cudaEventElapsedTime(&time_compute, compute, output);
    	cudaEventElapsedTime(&time_output, output, end);
	
	
	std::cout << "Time taken to load the dataset in the GPU memory: " << time_load << " ms." << std::endl;
	std::cout << "Time taken to compute (input binarization + bmm + binarized RELU): " << time_compute << " ms." << std::endl;
	std::cout << "Time taken to send the output to the host: " << time_output << " ms." << std::endl;



	cudaEventDestroy(load_input);
    	cudaEventDestroy(compute);
    	cudaEventDestroy(output);
    	cudaEventDestroy(end);


	return(out);
}
