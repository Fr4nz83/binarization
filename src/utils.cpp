//
// Created by cosimo on 14/01/20.
//

#include <stdio.h>
#include <stdlib.h>
#include <cmath>

//#include <time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <vector>
#include "utils.h"
#include <algorithm>
#include <random>
#include <set>
#include "./cnpy/cnpy.h"
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <vector>
#include <assert.h>
#include <filesystem>


#define ALIGN 64
using namespace cnpy;
using namespace std;
//using json = nlohmann::json;
namespace fs = std::filesystem;


DenseMatrix load_layer(std::string name)
{   
    std::string w_name;
    w_name= name + "/fc1.weight.npy";
    
    NpyArray h_npy = npy_load(w_name);
    int32_t height = h_npy.shape[0];
    std::cout << "height w (hidden units)" << height << "\n";
    
    int32_t width = h_npy.shape[1];
    std::cout<<"width (features)" << width << "\n";
    
    float* tmp = h_npy.data<float>();
    std::vector<float> h(tmp, tmp + height * width);
  
    
    // std::cout<<"bias"<<"\n";
    std::string b_name = name + "/fc1.bias.npy";
    NpyArray b_npy = npy_load(b_name);
    tmp = b_npy.data<float>();
    std::vector<float> b(tmp, tmp + height);
    printf("Weight size: %lux%lu,  Bias size: %lu  \n", h_npy.shape[0], h_npy.shape[1], b_npy.shape[0]);
    
    return(DenseMatrix(height, width, h, b));
}

void load_layer_weights(std::string name, vector<float>& h, vector<float>&  b, int& ext_w, int & ext_h){
    std::string w_name;
    w_name= name+".weight.npy";
    // std::cout<<"name "<<w_name<<"\n";
    
    NpyArray h_npy = npy_load(w_name);
    int height = h_npy.shape[0];
    std::cout << "height " << height << "\n";
    
    int width = h_npy.shape[1];
    std::cout<<"width "<<width<<"\n";
    
    h.resize(height*width);
    for(int i=0; i<height; i++){
        for (int j=0; j< width; j++){
            h[j+i*width] = h_npy.data<float>()[i*width+j];
        }
    }
    b.resize(height);
    
    // std::cout<<"bias"<<"\n";
    std::string b_name;
    b_name=name +".bias.npy";
    NpyArray b_npy = npy_load(b_name);
    height = b_npy.shape[0];
    for (int i= 0; i < height ; i++){
        b[i] = b_npy.data<float>()[i];
    }
    // printf("Weight size: %lux%lu,  Bias size: %lu  \n", h_npy.shape[0], h_npy.shape[1], b_npy.shape[0]);
    ext_h = height;
    ext_w = width;
}


vector<SparseMatrix> load_sparse_csr_weights_from_json(std::string folder_path) {


    vector<string> layers_name;

    for (const auto & entry : fs::directory_iterator(folder_path)) {
        string file_name = entry.path();
        file_name = file_name.substr(file_name.find_last_of("/")+1 ,file_name.size()-1 );
        string layer = file_name.substr(0, file_name.find("/"));
        if (layer.rfind("fc", 0)==0){
            //cout<<layer.substr(layer.find_first_of("."), layer.size()-1)<<" \n";
            if (layer.substr(layer.find_first_of("."), layer.size()-1) == ".weight.npy"){//avoid to add for weight and bias
                //cout<<layer.substr(0, layer.find_first_of("."))<<"\n";
                layers_name.push_back(layer.substr(0, layer.find_first_of(".")));
            }

        }


    }
    vector<SparseMatrix> sparse_weights;
    sort(layers_name.begin(), layers_name.end());

    for (const string layer : layers_name) {
        string weight_path = folder_path + "/" + layer;
        //cout << weight_path << "\n";

        int width, height;
        vector<float> h;
        vector<float> b;
        load_layer_weights(weight_path, h, b, width, height);
        vector<float> values;
        vector<int> columns;
        vector<int> rowIndex;
        float density;
        int nnz = 0;
        //rowIndex.push_back(0);
        for (size_t i =0; i<h.size(); i++){
            if(i%width ==0){
                rowIndex.push_back(nnz);
            }

            //if (h[i]!=0){
            if(abs(h[i])> 1e-6){
                values.push_back(h[i]);
                columns.push_back(i%width);
                nnz++;

            }

        }
        rowIndex.push_back(nnz);

        density = (float)values.size()/(height*width);



        SparseMatrix m = SparseMatrix(height, width, values, columns, rowIndex, b);
        sparse_weights.push_back(m);



    }
    return sparse_weights;
}


vector<DenseMatrix> load_dense_weights(string folder_path){
    vector<string> layers_name;

    for (const auto & entry : fs::directory_iterator(folder_path)) {
        string file_name = entry.path();
        file_name = file_name.substr(file_name.find_last_of("/")+1 ,file_name.size()-1 );
        string layer = file_name.substr(0, file_name.find("/"));
        if (layer.rfind("fc", 0)==0){
            //cout<<layer.substr(layer.find_first_of("."), layer.size()-1)<<" \n";
            if (layer.substr(layer.find_first_of("."), layer.size()-1) == ".weight.npy"){//avoid to add for weight and bias
                //cout<<layer.substr(0, layer.find_first_of("."))<<"\n";
                layers_name.push_back(layer.substr(0, layer.find_first_of(".")));
            }

        }


    }
    vector<DenseMatrix> weights;
    sort(layers_name.begin(), layers_name.end());



    for (const string layer : layers_name) {
        string weight_path = folder_path + "/" + layer;

        int width, height;
        vector<float> h;
        vector<float> b;
        load_layer_weights(weight_path, h, b, width, height);
        DenseMatrix m  = DenseMatrix(height, width, h, b);
        weights.push_back(m);
    }

    return weights;

}

float* stack_bias_verically(int M, int N, float* bias)
{
	cout << "Dimensione stacked bias: " << M*N << "(" << M << "x" << N << ")" << endl;
    auto stacked_bias = new float[M*N];
    for (int i=0; i<M; i++){
        for(int j=0; j<N; j++) {
            stacked_bias[i * N + j] = bias[i];
        }
    }
    return stacked_bias;
}

void random_init_dense(float * vect, int size) {

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> float_dist(-5, 5);
    for (int i = 0; i < size; i++) {
        vect[i] = (float) float_dist(gen);
    }
}

void init_sparse_matrix(float* values, int * columns, int * rowIndex, int nnz, int rows, int cols)
{
    int i;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> float_dist(0, 10);
    std::uniform_int_distribution<> dis(0, cols-1);
    //Matrix A
    for( i = 0; i < nnz; i++ )
        values[i] = (float) float_dist(gen);
    for( i = 0; i <nnz; i++ )//numeri interi tra 0 e cols
        columns[i] =(int) dis(gen);
    rowIndex[0] = 0;
    int step = nnz/rows;

    if (step !=0){
        for( i = 1; i < rows + 1; i++ )//posizionare nnz elementi
            rowIndex[i] = rowIndex[i - 1] + step;

        rowIndex[rows]+=nnz%rows;
    }
    else
    {
        for( i =1; i < nnz; i++)
            rowIndex[i] = rowIndex[i-1] + 1;
    }


}

float generate_sparsity( std::mt19937 gen, float min_sparsity, float max_sparsity)
{
    std::uniform_real_distribution<> float_dist(min_sparsity, max_sparsity);

    float sparsity = float_dist(gen);
    return sparsity;
}

std::vector<int> generate_random_offests(int height, int offset_size, std::mt19937 gen)
{
    std::vector<int> offsets;

    std::uniform_int_distribution<> dis(0, height-1);
    for (int j=0; j< offset_size; j++){
        offsets.push_back(dis(gen));
    }
    //sort offsets
    std::set<int> myset(offsets.begin(), offsets.end());    //erase duplicates
    offsets.resize(myset.size());
    std::copy(myset.begin(), myset.end(), offsets.begin());

    //sequentially add offsets to match the required number
    for (int j =0; j < offsets.size() && offsets.size() < offset_size; j++) {
        if (std::find(offsets.begin(), offsets.end(), j) == offsets.end()) {

            offsets.push_back(j);
        }
    }

    std::sort( offsets.begin(), offsets.end() );

    return offsets;

}

void random_init_striped_sparse(int height,
								int width,
								int n_stripes,
								float min_sparsity,
								float max_sparsity,std::vector<float*> & stripes,
								std::vector<std::vector<int>> &all_offsets,
								std::vector<float> &sparsities)
{
    int striped_width = width / n_stripes;
    for (int i=0; i< n_stripes; i++){

        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        float sparsity = min_sparsity;
        //float sparsity = generate_sparsity(gen, min_sparsity, max_sparsity);
        sparsities.push_back(sparsity);
        int offset_size = height - (int) (sparsity * (float)height);

        std::vector<int> offsets = generate_random_offests(height, offset_size, gen);
        all_offsets.push_back(offsets);
        auto *values = new float  [striped_width * offset_size];
        //auto values = (float*)mkl_malloc(striped_width*offset_size*sizeof(float), ALIGN);
        std::uniform_real_distribution<> float_dist(0, 10);
        for (int k=0; k< striped_width*offset_size; k++){
            values[k] = float_dist(gen);
        }
        stripes.push_back(values);
    }

}

int check_equality (float* a, float *b, int height, int width, float toll){
    int count_diff=0;
    float diff = 0;
    for (int i=0; i< height; i++) {
        for (int j = 0; j < width; j++) {
            diff = a[i * width + j] - b[i * width + j];
            if (std::abs(diff)> toll) {
                count_diff++;

                cout<< "Position: "<<i<<" , "<<j<<"\n";
            }
        }
    }
    if (count_diff!=0) std::cout<<"Number of different values : "<<count_diff<<"\n";
    assert(count_diff==0);
    return count_diff;

}

int check_equality (double* a, double *b, int height, int width, float toll){
    int count_diff=0;
    float diff = 0;
    for (int i=0; i< height; i++) {
        for (int j = 0; j < width; j++) {
            diff = a[i * width + j] - b[i * width + j];
            if (std::abs(diff)> toll) {
                count_diff++;
            }
        }
    }
    if (count_diff!=0) std::cout<<"Number of different values : "<<count_diff<<"\n";
    assert(count_diff==0);
    return count_diff;

}

float * get_dense_matrix_from_hybrid_vertical_stripes(int n_stripes,
													  int width,
													  int height,
													  int stripes_width,
													  std::vector<int> offset_sizes,
													  std::vector<float*> stripes,
													  std::vector<std::vector<int>> all_offsets)
{
    //float * dense_matrix = (float *)mkl_malloc(sizeof(float)*width*height, ALIGN) ;
    float * dense_matrix = new float [width*height] ;
    int row;
    for (int i=0; i< n_stripes; i++){
        for (std::vector<int>::size_type k=0; k<all_offsets[i].size(); k++){
            row = all_offsets[i][k];
            for(int j=0; j< stripes_width ; j++){
                    dense_matrix[row * width + i*stripes_width + j ] = stripes[i][k*stripes_width + j];
                }

        }
    }
    return dense_matrix;
}


void print_matrix(float *matrix, int height, int width)
{
    for (int i=0; i< height; i++)
    {
    	std::cout << "Line " << i << ": ";
        for (int j=0; j< width; j++)
        {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << endl;
    }
}

void print_matrix(double *matrix, int height, int width)
{
    for (int i=0; i< height; i++){
        for (int j=0; j< width; j++){
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout<<"\n";

    }
}

SparseMatrix convert_dense_to_sparse(DenseMatrix original_m)
{
    vector<float> values;
    vector<int> columns;
    vector<int> rowIndex;
    float density;
    int nnz = 0;

    for (size_t i =0; i<original_m.weights.size(); i++)
    {
        if((i % original_m.width) == 0)
        {
            rowIndex.push_back(nnz);
        }

        if (original_m.weights[i] != 0)
        {
            values.push_back(original_m.weights[i]);
            columns.push_back(i%original_m.width);
            nnz++;
        }

    }
    rowIndex.push_back(nnz);
    SparseMatrix m = SparseMatrix(original_m.height,
    							  original_m.width,
								  values,
								  columns,
								  rowIndex,
								  original_m.bias);

    return(m);
}

void clear_cache(int cache_size)
{
    //xor, and: nano /sys/devices/system/cpu/cpu0/cache/index0/size 32K

    const int size = cache_size * 10;
    char *c = (char *) malloc(size);
    for (int i = 0; i < 0xffff; i++)
        for (int j = 0; j < size; j++)
            c[j] = i * j;
}

void transpose_matrix(float *src, float *dst, const int N, const int M)
{
	std::cout << "Transposing matrix..." << std::endl;
	for(int n = 0; n<N*M; n++)
	{
		int i = n/N;
		int j = n%N;
		dst[n] = src[M*j + i];
	}
}
