/** @file mnist_mlp.cu
 *  @brief A 4-layer MLP for MNIST.
 *  @author Ang Li (PNNL)
 *
*/

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "utility.h"
#include "sbnn32_param.h"
#include "sbnn64_param.h"
#include "sbnn32.cuh"
#include "sbnn64.cuh"
#include "data.h"

#include "cnpy.h"
#include "generator.h"


using namespace cooperative_groups;
using namespace std;


// Nuovo modello per le NN...
__global__ void test(float* src, float* dest, int N)
{   
    grid_group grid = this_grid();
    thread_block tb = this_thread_block();
    
    const int num_blocks = grid.num_blocks();
    const int num_threads = grid.num_threads();
    const int tid = grid.thread_rank();
    const int tid_block = tb.thread_rank();
    
    if(tid == 0) printf("Num blocks: %d\n", num_blocks);
    if(tid == 0) printf("Num threads: %d\n", num_threads);
    
    if(tid == 0) dest[0] = 0;
    grid.sync();
    
    
    float val = 0;
    for(int i = tid; i < N; i += num_threads) val += src[i];
    tb.sync();
    
    
    // Perform sum-reduction.
    typedef cub::BlockReduce<int, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float aggregate = BlockReduce(temp_storage).Sum(val);
    
    
    if(tid_block == 0) atomicAdd(dest, aggregate);
}


using namespace cnpy;



int main_new();



int main()
{
	return main_new();
	// return main_old();
}




int main_new()
{
    cout << "Using BTSC-32\n";

    //=============== Configuration =================
    int dev = 0;
    cudaSetDevice(dev);



    //================ Set Network =================
    
    // *** Input Layer ***
    // NOTA: questo layer si aspetta una matrice di input in formato row-major con #righe == #docs e #colonne == #features.
    //cout << "Initializing the input layer..." << std::endl;
    //In32LayerParam* bin = new In32LayerParam("Fin", batch, n_feats);
    //In32LayerParam* bin_gpu = bin->initialize(images);
    
    // *** Fc1 Layer ***
    // NOTA: questo layer si aspetta una matrice dei pesi in formato row-major con #righe == #features e #colonne == #hidden_units. 
    //cout << "Initializing the weight layer..." << std::endl;
    //Fc32LayerParam* bfc1 = new Fc32LayerParam("Fc1", batch, n_feats, n_hidden); 
    //Fc32LayerParam* bfc1_gpu = bfc1->initialize(config_file, bin->get_output_gpu());
    
    

    //================ Setup Kernel =================
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, test, numThreads, 0);
    cout << "Number blocks per SM: " << numBlocksPerSm << std::endl;
    
    
    int N = 1000000;
    thrust::host_vector<float> t_h(N, 1);
    thrust::device_vector<float> t_d(t_h); float *t_d_ptr = thrust::raw_pointer_cast(t_d.data());
    thrust::device_vector<float> d_res(1, 1); float *d_res_ptr = thrust::raw_pointer_cast(d_res.data());
    std::cout << "Risultato PRE: " << d_res[0] << std::endl;
	 
	 
   
    
    
    void* args[] = {&t_d_ptr, &d_res_ptr, &N};
    cudaEvent_t start, end_load, end_ops;
    cudaEventCreate(&start);
    cudaEventCreate(&end_load);
    cudaEventCreate(&end_ops);
    double tot_time = 0, tot_time_comp = 0;
    uint32_t rounds = 1;
    for(uint32_t r = 0; r < rounds; r++)
    {
    	    //std::cout << "Round " << r << std::endl;
    
            float ms_init = 0;
    	    float ms_comp = 0;
    
	    
	    cudaEventRecord(end_load);
	    cudaLaunchCooperativeKernel((void*) test, 
	    				numBlocksPerSm * deviceProp.multiProcessorCount, 
	    				numThreads, 
	    				args);
	    cudaEventRecord(end_ops);
	    
	    
	    std::cout << "Risultato POST: " << d_res[0] << std::endl;
	    exit(1);
	    
	    
	    // float* res = bfc1->download_full_output();
	    // free(res);
	    
	    
	    cudaEventSynchronize(end_ops);
	    cudaEventElapsedTime(&ms_init, start, end_load);
    	    cudaEventElapsedTime(&ms_comp, end_load, end_ops);
    
            /*printf("\n============================\n");
            printf("Round: %d\n", r);
            printf("Init time: %.6f ms.\n", ms_init);
            printf("Comp time: %.6f ms.\n", ms_comp);
            printf("Tot time: %.6f ms.\n", ms_init + ms_comp);
            printf("============================\n");*/
            
            tot_time += ms_init + ms_comp;
            tot_time_comp += ms_comp;


	    //================ Output =================   
	    
	    // DEBUG: stampa risultati primo layer.
	    /*printf("\n============================\n");
	    printf("Print results\n");
	    float* res = bfc1->download_full_output();
	    for(uint32_t i = 0; i < batch; i++)
	    {
	    	cout << i << ": ";
	    	for(uint32_t j = 0; j < n_hidden; j++)
	    	{
	    		cout << res[i * n_hidden + j] << " ";
		}
		cout << endl;
	    }
	    printf("============================\n");*/
    }
    
    std::cout << "Average time load input + computation over " << rounds << " rounds: " << tot_time / rounds << " ms." << std::endl;
    // std::cout << "Average time load input + computation per element in the batch (tot: " << batch << ") over " << rounds << " rounds: " << tot_time * 1000 / rounds / batch << " us." << std::endl;
    std::cout << "Average time computation over " << rounds << " rounds: " << tot_time_comp / rounds << " ms." << std::endl;
    // std::cout << "Average time computation per doc (tot: " << batch << " docs) over " << rounds << " rounds: " << tot_time_comp * 1000 / rounds / batch << " us." << std::endl;
    std::cerr << tot_time_comp / rounds;


    //================ Release =================
    // delete bin;
    // delete bfc1;

    return 0;
}
