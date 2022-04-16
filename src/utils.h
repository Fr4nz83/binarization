//
// Created by cosimo on 14/01/20.
//

#ifndef HYBRIDSPARSEMULTIPLICATION_UTILS_H
#define HYBRIDSPARSEMULTIPLICATION_UTILS_H

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <vector>


using namespace std;


struct SparseMatrix
{
    int height;
    int width;
    std::vector<float> values;
    std::vector<int> columns;
    std::vector<int> rowIndex;
    std::vector<float> bias;
    float * stacked_bias;

    SparseMatrix(int M_,
    			 int K_,
				 std::vector<float> &values_,
				 std::vector<int> &columns_,
				 std::vector<int> &rowIndex_,
				 std::vector<float> &bias_ )
    {
        height = M_;
        width = K_;
        values = values_;
        columns = columns_;
        rowIndex = rowIndex_;
        bias = bias_;
        stacked_bias = 0;

        //sparse_status_t status = mkl_sparse_s_create_csr(csrA, SPARSE_INDEX_BASE_ZERO, height, width, rowIndex.data(), rowIndex.data() + 1, columns.data(), values.data());
        //
        // cout<<"Status"<<status<<"\n";
    }
};

struct DenseMatrix
{
    int height;
    int width;
    std::vector<float> weights;
    std::vector<float> bias;
    float* stacked_bias;
    DenseMatrix(int height_,
    			int width_,
				std::vector<float> & weights_,
				std::vector<float> bias_ )
    {
        height =height_;
        width= width_;
        weights = weights_;
        bias = bias_;
        stacked_bias = 0;
    }
};



void random_init_dense(float * vect, int size);

void init_sparse_matrix(float* values, int * columns, int * rowIndex, int nnz, int rows, int cols);

void random_init_striped_sparse(int height, int width, int n_stripes, float min_sparsity, float max_sparsity,std::vector<float*> & stripes, std::vector<std::vector<int>> &all_offsets, std::vector<float> &sparsities);

float* get_dense_matrix_from_hybrid_vertical_stripes(int n_stripes, int width, int height, int stripes_width, std::vector<int> offset_sizes, std::vector<float*> stripes, std::vector<std::vector<int>> all_offsets);
//float* measure_elapsed_time_sparse_striped(float* dense_matrix, int M, int K, int N, int n_stripes, int stripes_width, std::vector<float> sparsities, std::vector<std::vector<int>> all_offsets, std::vector<float *> stripes, int n_run);
//float* measure_elapsed_time_sparse_striped2(float* dense_matrix, int M, int K, int N, int n_stripes, int stripes_width, std::vector<float> sparsities, std::vector<std::vector<int>> all_offsets, std::vector<float *> stripes, int n_run);
//float* measure_elapsed_time_dense(int M, int K, int N, float * dense_matrix, float * input, int n_run);

void print_matrix(float *matrix, int height, int width);
void print_matrix(double *matrix, int height, int width);
int check_equality (float* a, float *b, int height, int width, float toll);
int check_equality (double* a, double *b, int height, int width, float toll);

DenseMatrix load_layer(std::string name);
void load_layer_weights(std::string name, std::vector<float>& h, std::vector<float>&  b, int& ext_w, int & ext_h);

vector<SparseMatrix> load_sparse_csr_weights_from_json(std::string folder_path);
vector<DenseMatrix> load_dense_weights(std::string folder_path);
float* stack_bias_verically(int M, int N, float*bias);
SparseMatrix convert_dense_to_sparse(DenseMatrix original_m);

void clear_cache(int cache_size);
//template <class T> int count_nonzero(const vector<T>& v);

void transpose_matrix(float *src, float *dst, const int N, const int M);


#endif //HYBRIDSPARSEMULTIPLICATION_UTILS_H
