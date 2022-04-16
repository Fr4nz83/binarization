/** @file mnist_mlp.cu
 *  @brief A 4-layer MLP for MNIST.
 *  @author Ang Li (PNNL)
 *
*/

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <cooperative_groups.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
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


__global__ void mnist_mlp32(In32LayerParam* bin, Fc32LayerParam* fc1)
{
    grid_group grid = this_grid();
    
    //========= Input ============
    In32Layer(bin);
    // In32LayerBatched(bin);
    grid.sync();
    
    //========== FC1 ============
    Fc32Layer(fc1);
    // Fc32LayerBatched(fc1);
    grid.sync();
}



using namespace cnpy;


int main()
{
    cout << "Using BTSC-32\n";

    //=============== Configuration =================
    int dev = 0;
    cudaSetDevice(dev);
    
    
    //=============== Load dataset =================
    NpyArray in_data = npy_load("d.npy"); // Load the dataset in its orginal format.
    const int32_t batch = in_data.shape[0]; // Number of observations
    const int32_t n_feats = in_data.shape[1];
    float *images = in_data.data<float>();
    cout << "Size dataset: " << batch << " elements, " << n_feats << " features" << endl;



    //================ Get Weights and biases =================
   
    NpyArray in_w = npy_load("w.npy"); // Load the weights.
    const int32_t n_hidden = in_w.shape[1];
    float *tmp_w = in_w.data<float>();
    std::vector<float> w(tmp_w, tmp_w + (in_w.shape[0] * in_w.shape[1]));
    cout << "Size weights: " << in_w.shape[0] << " n_feats, " << in_w.shape[1] << " n_hidden" << endl;
    cout << "Size vec weights: " << w.size() << endl;
    
    NpyArray in_b = npy_load("b.npy"); // Load the biases.
    float *tmp_b = in_b.data<float>();
    std::vector<float> b(tmp_b, tmp_b + in_b.shape[0]);
    cout << "Num biases: " << b.size() << endl;
    
    // Prepare the file all.txt, which is then used to initialize the first layer.
    // NOTE: It has to contain the weights followed by the biases, flattened and in row-major format. 
    w.insert(w.end(), b.begin(), b.end());
    write_array(w, "all.txt");
    FILE* config_file = fopen("./all.txt","r");



    //================ Set Network =================
    
    // *** Input Layer ***
    // NOTA: questo layer si aspetta una matrice di input in formato row-major con #righe == #docs e #colonne == #features.
    cout << "Initializing the input layer..." << std::endl;
    In32LayerParam* bin = new In32LayerParam("Fin", batch, n_feats);
    In32LayerParam* bin_gpu = bin->initialize(images);
    
    // *** Fc1 Layer ***
    // NOTA: questo layer si aspetta una matrice dei pesi in formato row-major con #righe == #features e #colonne == #hidden_units. 
    cout << "Initializing the weight layer..." << std::endl;
    Fc32LayerParam* bfc1 = new Fc32LayerParam("Fc1", batch, n_feats, n_hidden); 
    Fc32LayerParam* bfc1_gpu = bfc1->initialize(config_file, bin->get_output_gpu());
    
    
    
    //================ DEBUG =================
    /*cout << "righe (ovvero, righe singolo * size batch)': " << bin->input_height << endl;
    cout << "colonne': " << bin->input_width << endl;
    const float* ti = bin->input;
    cout << "Stampa input" << endl;
    for(uint32_t i = 0; i < bin->input_height; i++)
    {
    	cout << i << ": ";
    	for(uint32_t j = 0; j < bin->input_width; j++)
    	{
    		cout << ti[i * bin->input_width + j] << " ";
	}
	cout << endl;
    }
    
    
    cout << "Numero pesi: " << bfc1->weight_size() << endl;
    const float* tw = bfc1->weight;
    cout << "Stampa pesi" << endl;
    for(uint32_t i = 0; i < bfc1->weight_height; i++)
    {
    	cout << i << ": ";
    	for(uint32_t j = 0; j < bfc1->weight_width; j++)
    	{
    		cout << tw[i * bfc1->weight_width + j] << " ";
	}
	cout << endl;
    }
    
    cout << "Numero valori bn: " << bfc1->bn_size() << endl;
    const float* bn = bfc1->bn;
    cout << "Stampa bn" << endl;
    for(uint32_t i = 0; i < bfc1->bn_size(); i++)
	cout << bn[i] << " ";
    cout << endl;*/
    


    //================ Setup Kernel =================
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, mnist_mlp32, numThreads, 0);
    cout << "Number blocks per SM: " << numBlocksPerSm << std::endl;
    void* args[] = {&bin_gpu, &bfc1_gpu};
    

    cudaEvent_t start, end_load, end_ops;
    cudaEventCreate(&start);
    cudaEventCreate(&end_load);
    cudaEventCreate(&end_ops);
    double tot_time = 0, tot_time_comp = 0;
    uint32_t rounds = 100;
    for(uint32_t r = 0; r < rounds; r++)
    {
    	    std::cout << "Round " << r << std::endl;
    
            float ms_init = 0;
    	    float ms_comp = 0;
    
    
    	    cudaEventRecord(start);
	    bin->load_data(); // Method that is used to correctly measure the time needed to read the dataset.
	    //bin->reset_out_state(); // Method that (re)set the initial state of bin's output vector.
	    //bfc1->reset_out_state(); // Method that (re)set the initial state of bfc1's output vector.
	    
	    
	    cudaEventRecord(end_load);
	    cudaLaunchCooperativeKernel((void*) mnist_mlp32, numBlocksPerSm * deviceProp.multiProcessorCount, numThreads, args);
	    cudaEventRecord(end_ops);
	    
	    
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
    std::cout << "Average time load input + computation per element in the batch (tot: " << batch << ") over " << rounds << " rounds: " << tot_time * 1000 / rounds / batch << " us." << std::endl;
    std::cout << "Average time computation over " << rounds << " rounds: " << tot_time_comp / rounds << " ms." << std::endl;
    std::cout << "Average time computation per doc (tot: " << batch << " docs) over " << rounds << " rounds: " << tot_time_comp * 1000 / rounds / batch << " us." << std::endl;
    std::cerr << tot_time_comp / rounds;


    //================ Release =================
    delete bin;
    delete bfc1;

    return 0;
}
