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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "utility.h"
#include "sbnn32_param.h"
#include "sbnn32.cuh"
#include "data.h"
#include "dataset_reader.h"

#include "binarization.cuh"

#include "cnpy.h"



// *** FORWARD DECLARATIONS *** //
int main_new();



// *** MAIN *** //

int main()
{
	return main_new();
}


int main_new()
{
    using namespace cooperative_groups;
    
    
    cout << "Using BTSC-32\n";

    //=============== Device Configuration =================
    int dev = 0;
    cudaSetDevice(dev);



    //================ Set Network =================
    
    // *** Input Layer ***
    constexpr uint32_t image_height = 32, image_width = 32, image_channel = 3, test_batch = 10000;
    In32Conv32LayerParam* bconv1 = new In32Conv32LayerParam("Conv1",
    														image_height, image_width, // Image size
                											3, 3,     // Filter size
															3, 128,   // # input channels and # output channels
															test_batch, // Batch size
															1,1, // Stride offsets
															true, // same padding flag
															0,0,      // Pooling size
															false,	  // Transpose flag
															true);    //save residual flag
    
    // *** Fc1 Layer ***
    // NOTA: questo layer si aspetta una matrice dei pesi in formato row-major con #righe == #features e #colonne == #hidden_units. 
    //cout << "Initializing the weight layer..." << std::endl;
    //Fc32LayerParam* bfc1 = new Fc32LayerParam("Fc1", batch, n_feats, n_hidden); 
    //Fc32LayerParam* bfc1_gpu = bfc1->initialize(config_file, bin->get_output_gpu());
    
    

    //================ Setup parameters Kernels =================
    constexpr int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, calc_stats, numThreads, 0);
    cout << "Number blocks per SM with calc_stats kernel: " << numBlocksPerSm << std::endl;

    
    std::string cifar10_dir = "../dataset/data_batch_1.bin";
    auto set_images = ImgDatasetReader<image_height,image_width>::read_dataset_cifar10_float(cifar10_dir);
    std::cout << "Number of images: " << set_images.size() << std::endl;
    
    const uint32_t batch = set_images.size();
    uint32_t *images_test = (uint32_t*) malloc(batch * image_height * image_width * sizeof(uint32_t));
    uint32_t *image_labels_test = (uint32_t*) malloc(batch * sizeof(uint32_t));
    read_CIFAR10_raw(cifar10_dir, images_test, image_labels_test, batch);


    // TODO: verificare che il contenuto delle due letture sia uguale.
    exit(1);
    
    // TODO: non appena si capisce con quale formato vengono passati i dati al layer Input della convoluzione, adattare la classe
    //       ImgInLayer32 in maniera tale che prenda i dati nello stesso formato.
    std::vector<float> tst;
    ImgInLayer32 in_layer(tst);
    
    
    cudaEvent_t start, end_load, end_ops; 
    cudaEventCreate(&start); cudaEventCreate(&end_load); cudaEventCreate(&end_ops);
    double tot_time = 0, tot_time_comp = 0;
    constexpr uint32_t rounds = 1;
    for(uint32_t r = 0; r < rounds; r++)
    {
    	    std::cout << "Round " << r << std::endl;
    
            float ms_init = 0;
    	    float ms_comp = 0;
    
    
    	    // 1 - Transfer INPUT data from host to GPU...
	    cudaEventRecord(start);
	    ImgInLayer32 *in_layer_gpu = in_layer.initialize();
	    void* args[] = {&in_layer_gpu};
	    
	    
	    
	    // 2 - Compute some statistics concerning the input data.
	    cudaEventRecord(end_load);
	    cudaLaunchCooperativeKernel((void*) calc_stats, 
	    				numBlocksPerSm * deviceProp.multiProcessorCount, 
	    				numThreads, 
	    				args);
	    cudaEventRecord(end_ops);
	    
	    
	    // DEBUG.
	    std::cout << "Risultato mean: " << in_layer.get_mean() << std::endl;
	    std::cout << "Risultato variance: " << in_layer.get_variance() << std::endl;
	    // float* res = bfc1->download_full_output();
	    // free(res);
	    
	    
	    
	    // 3 - Execute the NN forward pass.
	    // TODO.
	    
	    
	    
	    cudaEventSynchronize(end_ops);
	    cudaEventElapsedTime(&ms_init, start, end_load);
    	    cudaEventElapsedTime(&ms_comp, end_load, end_ops);
    
            printf("\n============================\n");
            printf("Round: %d\n", r);
            printf("Load data time: %.6f ms.\n", ms_init);
            printf("Computation time: %.6f ms.\n", ms_comp);
            printf("Tot time: %.6f ms.\n", ms_init + ms_comp);
            printf("============================\n");
            
            tot_time += ms_init + ms_comp;
            tot_time_comp += ms_comp;
    }
    
    
    
    std::cout << "Average time load input + computation over " << rounds << " rounds: " << tot_time / rounds << " ms." << std::endl;
    // std::cout << "Average time load input + computation per element in the batch (tot: " << batch << ") over " << rounds << " rounds: " << tot_time * 1000 / rounds / batch << " us." << std::endl;
    std::cout << "Average time computation over " << rounds << " rounds: " << tot_time_comp / rounds << " ms." << std::endl;
    // std::cout << "Average time computation per doc (tot: " << batch << " docs) over " << rounds << " rounds: " << tot_time_comp * 1000 / rounds / batch << " us." << std::endl;
    std::cerr << tot_time_comp / rounds;



    //================ Release resources =================
    // TODO ...
    

    return 0;
}
