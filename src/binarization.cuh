#pragma once


#include <vector>

#include <cooperative_groups.h>
#include <cub/cub.cuh>


using namespace cooperative_groups;



class ImgInLayer32
{
public:
	
	// *** PUBLIC FIELDS *** //

	// Input fields.
        float *input;
        float scale, shift;
        float *d_input;
        
        // Fields for internal usage.
        int size_input;
        float *d_tmp_buff;
        float *d_mean, *d_variance;
        
        // Shadow object on GPU.
        ImgInLayer32* gpu;


	
	// *** PUBLIC CTORS \ DTOR *** //
	
	ImgInLayer32(std::vector<float>& input) : 
	input(input.data()), scale(1), shift(0), size_input(input.size()), gpu(0)
	{		
		// Allocate the regions of memory needed by the various fields.
		cudaMalloc((void**)&(this->d_input), this->size_input * sizeof(float));
		cudaMalloc((void**)&(this->d_tmp_buff), this->size_input * sizeof(float));
        	cudaMalloc((void**)&(this->d_mean), sizeof(float));
        	cudaMalloc((void**)&(this->d_variance), sizeof(float));
        	cudaMalloc((void**)&(this->gpu), sizeof(ImgInLayer32));
	}
	
	~ImgInLayer32()
	{
		cudaFree(this->d_input);
		cudaFree(this->d_tmp_buff);
		cudaFree(this->d_mean);
		cudaFree(this->d_variance);
		if(this->gpu) cudaFree(this->gpu);
	}
		
		
		
        // *** PUBLIC METHODS *** //
        
        float get_mean()
        {
        	float res;
        	cudaMemcpy(&res, this->d_mean, sizeof(float), cudaMemcpyDeviceToHost);
        	return(res);
        }
        
        float get_variance()
        {
        	float res;
        	cudaMemcpy(&res, this->d_variance, sizeof(float), cudaMemcpyDeviceToHost);
        	return(res);
        }
        
        ImgInLayer32* initialize()
        {        	
        	// Transfer the data from the CPU to the GPU.
        	cudaMemcpy(this->d_input, this->input, this->size_input * sizeof(float), cudaMemcpyHostToDevice);
        	cudaMemcpy(this->gpu, this, sizeof(ImgInLayer32), cudaMemcpyHostToDevice);
        	
        	return(this->gpu);
        }
};



// *** KERNELS AND DEVICE FUNCTIONS *** //

__device__ inline void sum_reduction(const grid_group& grid, float *input, int N, float *res)
{
    thread_block tb = this_thread_block();
    
    // Local variables
    const int num_threads = grid.num_threads();
    const int tid = grid.thread_rank();
    const int tid_block = tb.thread_rank();
    // const int warpID = num_threads / 32;
    // const int laneID = num_threads % 32;
    
    
    // Each thread accumulates the values it has to read. 
    float val = 0;
    for(int i = tid; i < N; i += num_threads) 
    	val += input[i];
    tb.sync(); // Thread-block sync.
    
    
    // Perform sum-reduction at thread-block level.
    // TODO: qui il numero di thread usato per la reduce e' hardcoded a 1024...vedere se si puo' rimuovere questa dipendenza (ad es. via templatizzazione).
    typedef cub::BlockReduce<float, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float aggregate = BlockReduce(temp_storage).Sum(val);
    
    
    // The first thread in each block adds atomically the sum computed within the block.
    if(tid_block == 0) atomicAdd(res, aggregate);
    grid.sync();
}

__device__ inline void compute_mean(const grid_group& grid, ImgInLayer32* in_layer)
{
    const int tid = grid.thread_rank();
    
    
    sum_reduction(grid, in_layer->d_input, in_layer->size_input, in_layer->d_mean);
    
    
    if(tid == 0) in_layer->d_mean[0] = in_layer->d_mean[0] / in_layer->size_input;
    grid.sync();
}

__device__ inline void compute_variance(const grid_group& grid, ImgInLayer32* in_layer)
{
    const int num_threads = grid.num_threads();
    const int tid = grid.thread_rank();
    
    
    // Prepare the buffer that will be used to compute the variance.
    const auto& N = in_layer->size_input;
    const float mean = in_layer->d_mean[0];
    for(int i = tid; i < N; i += num_threads)
    {
    	float tmp = in_layer->d_input[i] - mean; 
    	in_layer->d_tmp_buff[i] = tmp * tmp;
    }
    grid.sync();
    
    
    // Sum reduction of the variance.
    sum_reduction(grid, in_layer->d_tmp_buff, in_layer->size_input, in_layer->d_variance);
    if(tid == 0) in_layer->d_variance[0] = in_layer->d_variance[0] / in_layer->size_input;
    grid.sync();
}

__device__ inline void transform_input(const grid_group& grid, ImgInLayer32* in_layer)
{
    const int num_threads = grid.num_threads();
    const int tid = grid.thread_rank();
    
    
    // Prepare the buffer that will be used to compute the variance.
    constexpr float eps = 1e-8;
    const float mean = in_layer->d_mean[0];
    const float divisor = sqrt(in_layer->d_variance[0] + eps);
    const float scale = in_layer->scale;
    const float shift = in_layer->shift;
    
    
    const auto& N = in_layer->size_input;
    for(int i = tid; i < N; i += num_threads)
    {
    	const float tmp = (in_layer->d_input[i] - mean) / divisor; 
    	in_layer->d_tmp_buff[i] = scale * tmp + shift;
    }
    grid.sync();
}


// This kernel computes a few statistics.
__global__ void calc_stats(ImgInLayer32* in_layer)
{
    // Init object offering grid-wide primitives.
    grid_group grid = this_grid();
    const int tid = grid.thread_rank();
    
    
    // Initialize the variable which will hold the final sum reduction.    
    if(tid == 0) 
    {
    	in_layer->d_mean[0] = 0;
    	in_layer->d_variance[0] = 0;
    }
    grid.sync();
    
    
    // Compute the mean of the values...
    compute_mean(grid, in_layer);
    
    
    // Compute the variance...
    compute_variance(grid, in_layer);
    
    
    // Transform the original data according to the mean, variance, scale, and shift parameters.
    transform_input(grid, in_layer);
}
