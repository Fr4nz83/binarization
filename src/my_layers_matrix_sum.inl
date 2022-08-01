// *** FORWARD DECLARATIONS OF KERNELS USED BY THIS CLASS *** //

__global__ void Matrix_Sum(MatrixSumLayer* p);



// *** CTORS/DTOR DEFINITIONS *** //

MatrixSumLayer::MatrixSumLayer(const char* name,
				     	 	   const unsigned& data_width,
							   const unsigned& data_height) :
input1_gpu(NULL),
input2_gpu(NULL),
output_gpu(NULL),
input_width(data_width),
input_height(data_height),
gpu(NULL)
{
	strncpy(this->name, name, 8);
	std::cout << "Invoking constructor for " << this->name << std::endl;
}



// *** PUBLIC METHODS DEFINITIONS *** //

void MatrixSumLayer::release()
{
	std::cout << "Dealloc CUDA resources..." << std::endl;


	// Dealloc data space (may be NULL in case this layer is connected to other layers).
	if(this->input1_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->input1_gpu) );
		this->input1_gpu = NULL;
	}

	if(this->input2_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->input2_gpu) );
		this->input2_gpu = NULL;
	}

	if(this->output_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->output_gpu) );
		this->output_gpu = NULL;
	}

	if(this->gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->gpu) );
		this->gpu = NULL;
	}
}

MatrixSumLayer* MatrixSumLayer::ready()
{
	// Dealloc data space (may be NULL in case this layer is connected to other layers).
	if(this->input1_gpu == NULL || this->input2_gpu == NULL || this->input_size() == 0)
	{
		std::cout << "Input data has not been allocated/initialized on the GPU." << std::endl;
		exit(1);
	}

	// Allocate shadow copy of this instance on GPU.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->gpu), sizeof(MatrixSumLayer)));
	CUDA_SAFE_CALL(cudaMemcpy(this->gpu, this, sizeof(MatrixSumLayer), cudaMemcpyHostToDevice));


	// Return the pointer to the shadow copy (to be used within a kernel).
	return this->gpu;
}

void MatrixSumLayer::allocate_output_gpu()
{
	if(this->output_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->output_gpu) );
		this->output_gpu = NULL;
	}

	// Allocate space for output.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->output_gpu), this->output_bytes()));
}

void MatrixSumLayer::load_input_gpu(float* input1, float* input2)
{

	if(this->input1_gpu != NULL)
	{
		CUDA_SAFE_CALL(cudaFree(this->input1_gpu));
		this->input1_gpu = NULL;
	}

	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->input1_gpu), this->input_bytes()));
	CUDA_SAFE_CALL(cudaMemcpy(this->input1_gpu, input1, this->input_bytes(), cudaMemcpyHostToDevice));

	if(this->input2_gpu != NULL)
	{
		CUDA_SAFE_CALL(cudaFree(this->input2_gpu));
		this->input2_gpu = NULL;
	}

	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->input2_gpu), this->input_bytes()));
	CUDA_SAFE_CALL(cudaMemcpy(this->input2_gpu, input2, this->input_bytes(), cudaMemcpyHostToDevice));


	this->allocate_output_gpu();
}

void MatrixSumLayer::download_output_gpu(float* output)
{
	CUDA_SAFE_CALL(cudaMemcpy(output, this->output_gpu, this->output_bytes(), cudaMemcpyDeviceToHost));
}

void MatrixSumLayer::execute_layer()
{
	// 1 - Prepare the layer for execution.
	MatrixSumLayer* gpu_copy = this->ready();


	// 2 - Input binarization kernel execution.
	std::cout << "Matrix sum..." << std::endl;

	constexpr uint32_t WARP_SIZE = 32;
	const uint32_t NUM_BLOCKS = this->input_size() / WARP_SIZE + (this->input_size() % WARP_SIZE != 0);
	Matrix_Sum <<<NUM_BLOCKS, WARP_SIZE>>> (gpu_copy);
}



// *** CUDA KERNELS *** //

/**
 * @brief This kernel computes the element-wise sum between two matrices.
 */
__global__ void Matrix_Sum(MatrixSumLayer* p)
{
    const uint32_t num_threads_grid = blockDim.x * gridDim.x;
    const uint32_t start_address_thread = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t size_matrix = p->input_height * p->input_width;

    for(uint32_t address = start_address_thread; address < size_matrix; address += num_threads_grid)
    	p->output_gpu[address] = p->input1_gpu[address] + p->input2_gpu[address];
}
