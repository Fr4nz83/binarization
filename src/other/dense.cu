#include <iostream>

#include <random>
#include "./cnpy/cnpy.h"
#include "utils.h"
#include <cstdlib>
#include <chrono>

#include <boost/program_options.hpp>

#include "gpu_mat.h"


using namespace std;
using namespace cnpy;


/*
 * @brief This function invokes the GPU scorer.
 */
tuple<void*, double> forward_gpu(const int batch_size,
				 GPU_MatMul& gpu_scorer,
				 void* input,
				 const uint32_t& numFeats)
{
	// Score the docs on GPU
	// NOTE: this entails the execution of the first (sparse) layer plus the other dense layers that follow.
	void* output;
	output = gpu_scorer.score_GPU(batch_size, static_cast<float*>(input), numFeats);
    	return {output, gpu_scorer.get_time_compute()};
}



int main(int argc, char * argv[])
{
	std::string dataset_path;
	std::string model_path;
	uint32_t rounds;
	uint32_t batch_size;


	// *** 0 - Command line parsing *** //
	namespace po = boost::program_options;
	try
	{
		po::options_description options("Options");
		options.add_options()("help,h", "Print help messages");
		options.add_options()("dataset,d",
					  po::value<std::string>(&dataset_path)->required(),
					  "Input dataset in numpy format");
		options.add_options()("model,m",
					  po::value<std::string>(&model_path)->required(),
					  "DNN model in numpy format");
		options.add_options()("rounds,r",
					  po::value<uint32_t>(&rounds)->default_value(1),
					  "Number of test repetitions (default 1)");
		options.add_options()("batch,b",
					  po::value<uint32_t>(&batch_size)->default_value(0),
					  "Number of documents per batch (default is 0, i.e., batch size is the number of documents in the dataset)");

		// parse command line
		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, options), vm);

		// print help
		if (vm.count("help")) {
			std::cout << options << "\n";
			return EXIT_FAILURE;
		}
		// raise any error
		po::notify(vm);
	}
	catch (const po::error &ex)
    {
		std::cout << "Command line parsing error (empty or missing mandatory arguments!)" << std::endl;
		std::cout << ex.what() << '\n';
		return EXIT_FAILURE;
    }



    // *** 1 - Use the cnpy library to load a dataset saved with the np array format into a float array *** //
    cout << "Reading Dataset from: " << dataset_path << endl;

    NpyArray data = npy_load(dataset_path); // Load the dataset in its orginal format.
    float *dataset = data.data<float>();    // Allocate a float array in the heap and set its content to the values stored in the np array.

    // Extract the characteristics of the dataset.
    // NOTA: Il dataset e' una matrice (dataset_size x num_feats)
    int n_features = data.shape[1];
    int dataset_size = data.shape[0]; // Number of observations

    cout << "Dataset size: " << dataset_size << endl;
    cout << "# features: " << n_features << endl;


    // *** 2 - Read the DNN model *** //
    cout << "Reading the DNN model from: " << model_path << "\n";
    cout << "Loading the weights associated with the layers..." << endl;
    vector<DenseMatrix> denseWeights(1,load_layer(model_path));
    for(auto d = denseWeights.begin(); d != denseWeights.end(); ++d)
    {
		cout << "Weights for this layer -- height: " << d->height << ", width: " << d->width << endl;
		// print_matrix(d->weights.data(), d->height, d->width);
    }
    cout << endl;



    // *** 3 - Lettura del batch size, con conseguente calcolo di altri parametri interni *** //
    batch_size = batch_size ? batch_size : dataset_size;
    cout << "Number of documents that will be scored: " << batch_size << endl;



    // *** 4 - Alloca e scrivi in memoria il dataset in formato trasposto *** //
    // NOTA: il dataset trasposto sara' costituito da matrice avente dimensioni (num_feats x dataset_size)
    // float *input; cudaMallocHost((void**)&input, dataset_size * n_features * sizeof(float));
    float *input = new float[dataset_size * n_features];
    transpose_matrix(dataset, input, dataset_size, n_features);
    std::cout << "Trasposta calcolata!" << std::endl;

    // *** 4.2 - Initialize the GPU and the associated data structures. *** //
    GPU_MatMul gpu_scorer(denseWeights); // Allocate the class in charge of holding and using the DNN model on the GPU.


    // 5 - Scoring execution.
    double total_time = 0;
    void *complete_output = 0;
    cout << "Start keeping track of the execution time..." << std::endl;
    for (int j = 0; j < rounds; j++) // Effettua "n_run" runs sull'insieme dei documenti da scorare.
    {
		// Score the documents.
    	auto [output, time] = forward_gpu(batch_size,
					  gpu_scorer,
					  input,
					  n_features);

	if (j != (rounds - 1)) 
		free(output);
	else 
		complete_output = output;
    	
    	total_time += time;
    }
    cout<<"Avg. time spent computing per doc per round (input binarization, bmm, RELU + binarization): "<< total_time * 1000 / rounds / dataset_size << " μs/doc\n";
    cout<<"Avg. time spent computing per round (input binarization, bmm, RELU + binarization): "<< total_time / rounds << " ms\n";


    // 6 - Salva a disco i risultati dello scoring.
    // cnpy::npy_save("complete_output.npy", complete_output.data(), {complete_output.size()},"w");


    // Libera le risorse allocate nello heap.
    cudaFreeHost(input);
}
