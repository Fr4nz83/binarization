// *** CTORS/DTOR DEFINITIONS *** //

TransposeFullPrecLayer::TransposeFullPrecLayer(const char* name,
											   const unsigned& data_width,
											   const unsigned& data_height,
											   const unsigned& data_channels) :
size_batch(0),
input_gpu(NULL),
output_gpu(NULL),
input_width(data_width),
input_height(data_height),
input_channels(data_channels),
gpu(NULL)
{
	strncpy(this->name, name, 8);
	std::cout << "Invoking constructor for " << this->name << std::endl;
}

// *** PUBLIC METHODS DEFINITIONS *** //

void TransposeFullPrecLayer::release()
{
	std::cout << "Dealloc CUDA resources..." << std::endl;


	this->size_batch = 0;


	// Various deallocs.
	if(this->input_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->input_gpu) );
		this->input_gpu = NULL;
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

TransposeFullPrecLayer* TransposeFullPrecLayer::ready()
{
	// Dealloc data space (may be NULL in case this layer is connected to other layers).
	if(this->input_gpu == NULL || this->input_size() == 0 || this->output_gpu == NULL)
	{
		std::cout << "Input data has not been allocated/initialized on the GPU." << std::endl;
		exit(1);
	}

	// Allocate shadow copy of this instance on GPU.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->gpu), sizeof(TransposeFullPrecLayer)));
	CUDA_SAFE_CALL(cudaMemcpy(this->gpu, this, sizeof(TransposeFullPrecLayer), cudaMemcpyHostToDevice));


	// Return the pointer to the shadow copy (to be used within a kernel).
	return this->gpu;
}

void TransposeFullPrecLayer::load_input_gpu(float* input, unsigned size_batch)
{
	this->size_batch = size_batch;

	// Allocate and load input.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->input_gpu), this->input_bytes()));
	CUDA_SAFE_CALL(cudaMemcpy(this->input_gpu, input, this->input_bytes(), cudaMemcpyHostToDevice));
}

void TransposeFullPrecLayer::allocate_output_gpu()
{
	// Allocate space for output.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->output_gpu), this->input_bytes()));
}

void TransposeFullPrecLayer::download_output_gpu(float* output)
{
	CUDA_SAFE_CALL(cudaMemcpy(output, this->output_gpu, this->input_bytes(), cudaMemcpyDeviceToHost));
}



// *** CUDA KERNELS *** //

/**
 * @brief This kernel processes batch-normalizes the channels of a set of images.
 *
 * @note The kernel assumes that the dataset is stored in NCHW format.
 */
// TODO: this kernel will have to be a __device__ function at some point.
__global__ void TransposeFPLayer(TransposeFullPrecLayer* p)
{
	// General properties for this kernel.
	constexpr uint8_t WARPSIZE = 32;
	const uint8_t lane_id = threadIdx.x % WARPSIZE;
	const uint32_t& block_id = blockIdx.x;
	const uint32_t& num_blocks_grid = gridDim.x;


	// Shared memory buffer for a single warp.
	__shared__ float sub_m[WARPSIZE][WARPSIZE + 1];


	// General properties of an image.
	const uint32_t height = p->input_height;
	const uint32_t width = p->input_width;
	const uint32_t blocks_rows_img = height / WARPSIZE + (height % WARPSIZE != 0);
	const uint32_t blocks_columns_img = width / WARPSIZE + (width % WARPSIZE != 0);
	const uint32_t num_blocks_per_img = blocks_rows_img * blocks_columns_img;
	const uint32_t img_size = height * width;
	const uint32_t num_channels = p->input_channels;


	// Each block (warp) processes an image at a time.
	for(uint32_t id_img = block_id; id_img < p->size_batch; id_img += num_blocks_grid)
	{
		// Process each color dataset-wise.
		for(uint32_t c = 0; c < num_channels; c++)
		{
			// Compute the memory offset where the values associated with a given image and channel block start.
			const uint32_t offset_img = id_img * (num_channels * img_size) + (c * img_size);

			// Here the threads in a warp read a sub-matrix of max 32x32 values.
			// The sub-matrix gets transposed via the shared memory buffer.
			for(uint32_t block = 0; block < num_blocks_per_img; block++)
			{
				// Determine the indices of the block of rows and columns currently considered.
				const uint32_t idx_block_rows = block / blocks_columns_img;
				const uint32_t idx_block_columns = block % blocks_columns_img;

				// Compute the indices of the first row and column of this block.
				const uint32_t start_row = idx_block_rows * WARPSIZE;
				const uint32_t end_row = min(start_row + WARPSIZE, height);

				// Determine the indices of the last row and column of this block.
				const uint32_t start_column = idx_block_columns * WARPSIZE;
				const uint32_t end_column = min(start_column + WARPSIZE, width);


				// 1 - Write a row-major sub-matrix of up to 32x32 elements to shared memory.
				#pragma unroll WARPSIZE
				for(uint32_t row = start_row; row < end_row; row++)
				{
					// Read a single row of up to 32 elements.
					// In the shared mem buffer each row represents a sub-column which values
					// were read from the thread having ID "lane_id" within the warp.
					const uint32_t read_offset_tid = offset_img + (row * width) + (start_column + lane_id);
					if(start_column + lane_id < end_column)
						sub_m[lane_id][row % WARPSIZE] = p->input_gpu[read_offset_tid];
				}


				// 2 - Now write the matrix transposed from shared memory to global memory.
				#pragma unroll WARPSIZE
				for(uint32_t column = start_column; column < end_column; column++)
				{
					// Read a single row of up to 32 elements.
					// In the shared mem buffer each row represents a sub-column which values
					// were read from the thread having ID "lane_id" within the warp.
					const uint32_t write_offset_tid = offset_img + (column * height) + (start_row + lane_id);
					if(start_row + lane_id < end_row)
						p->output_gpu[write_offset_tid] = sub_m[column % WARPSIZE][lane_id];
				}
			}
		}
	}
}
