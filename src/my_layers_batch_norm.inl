// *** FORWARD DECLARATIONS OF CUDA KERNELS USED BY THIS CLASS *** //

__global__ void BNFPLayer(BatchNormFullPrecLayer* p);



// *** CTORS/DTOR DEFINITIONS *** //

BatchNormFullPrecLayer::BatchNormFullPrecLayer(const char* name,
											   const unsigned& data_width,
											   const unsigned& data_height,
											   const unsigned& data_channels,
											   const float* scale,
											   const float* shift) :
size_batch(0),
input_gpu(NULL),
input_width(data_width),
input_height(data_height),
input_channels(data_channels),
gpu(NULL)
{
	strncpy(this->name, name, 8);
	std::cout << "Invoking constructor for " << this->name << std::endl;

	// Load the scaling scalars (1 per channel) into the GPU.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->scale_gpu), data_channels * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(this->scale_gpu, scale, data_channels * sizeof(float), cudaMemcpyHostToDevice));

	// Load the shift scalars (1 per channel) into the GPU.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->shift_gpu), data_channels * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(this->shift_gpu, shift, data_channels * sizeof(float), cudaMemcpyHostToDevice));
}

// *** PUBLIC METHODS DEFINITIONS *** //

void BatchNormFullPrecLayer::release()
{
	std::cout << "Dealloc CUDA resources..." << std::endl;


	this->size_batch = 0;

	// Dealloc data space (may be NULL in case this layer is connected to other layers).
	if(this->input_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->input_gpu) );
		this->input_gpu = NULL;
	}

	// Dealloc scale factors vector.
	if(this->scale_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->scale_gpu) );
		this->scale_gpu = NULL;
	}

	// Dealloc shift factors vector.
	if(this->shift_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->shift_gpu) );
		this->shift_gpu = NULL;
	}
}

BatchNormFullPrecLayer* BatchNormFullPrecLayer::ready()
{
	// Dealloc data space (may be NULL in case this layer is connected to other layers).
	if(this->input_gpu == NULL || this->input_size() == 0)
	{
		std::cout << "Input data has not been allocated/initialized on the GPU." << std::endl;
		exit(1);
	}

	// Dealloc scale factors vector.
	if(this->scale_gpu == NULL)
	{
		std::cout << "Scale factors have not been copied to the GPU." << std::endl;
		exit(1);
	}

	// Dealloc shift factors vector.
	if(this->shift_gpu == NULL)
	{
		std::cout << "Shift factors have not been copied to the GPU." << std::endl;
		exit(1);
	}


	// Allocate shadow copy of this instance on GPU.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->gpu), sizeof(BatchNormFullPrecLayer)));
	CUDA_SAFE_CALL(cudaMemcpy(this->gpu, this, sizeof(BatchNormFullPrecLayer), cudaMemcpyHostToDevice));


	// Return the pointer to the shadow copy (to be used within a kernel).
	return this->gpu;
}

void BatchNormFullPrecLayer::load_input_gpu(const unsigned& size_batch, const std::vector<void*>& input)
{
	this->size_batch = size_batch;

	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->input_gpu), this->input_bytes()));
	CUDA_SAFE_CALL(cudaMemcpy(this->input_gpu, input[0], this->input_bytes(), cudaMemcpyHostToDevice));

	this->allocate_output_gpu();
}

void BatchNormFullPrecLayer::download_output_gpu(void* output)
{
	CUDA_SAFE_CALL(cudaMemcpy(output, this->input_gpu, this->output_bytes(), cudaMemcpyDeviceToHost));
}

void BatchNormFullPrecLayer::execute_layer()
{
	constexpr uint32_t WARP_SIZE = 32;

	// 1 - Prepare the layer for execution.
	BatchNormFullPrecLayer* gpu_copy = this->ready();

	// 2 - Execute layer.
	BNFPLayer <<<this->size_batch, WARP_SIZE>>> (gpu_copy);
}



// *** CUDA KERNELS *** //

/**
 * @brief This kernel processes batch-normalizes the channels of a set of images.
 *
 * @note The kernel assumes that the dataset is stored in NCHW format.
 */
// TODO: this kernel will have to be a __device__ function at some point.
__global__ void BNFPLayer(BatchNormFullPrecLayer* p)
{
	constexpr uint8_t WARPSIZE = 32;

	const uint8_t warp_id = threadIdx.x / WARPSIZE;
	const uint8_t lane_id = threadIdx.x % WARPSIZE;
	const uint32_t warps_block = blockDim.x / WARPSIZE;
	const uint32_t& block_id = blockIdx.x;
	const uint32_t& num_blocks = gridDim.x;
	const uint32_t num_img_per_grid = num_blocks * warps_block;


	const uint32_t img_size = p->input_width * p->input_height;
	const uint32_t num_channels = p->input_channels;


	// Process each color dataset-wise.
	for(uint32_t c = 0; c < num_channels; c++)
	{
		// Read the scale and shift factors associated with the channel currently considered.
		const float scale = p->scale_gpu[c];
		const float shift = p->shift_gpu[c];

		// Each warp processes an image per loop.
		for(uint32_t id_img = block_id * warps_block + warp_id; id_img < p->size_batch; id_img += num_img_per_grid)
		{
			// Compute the memory offset where the values associated with a given image and channel block start.
			const uint32_t offset_img = id_img * (num_channels * img_size) + (c * img_size);

			// Here threads in a warp use coalescing while reading and then updating the values.
			for(uint32_t i = lane_id; i < img_size; i += WARPSIZE)
				p->input_gpu[offset_img + i] = scale * p->input_gpu[offset_img + i] + shift;
		}
	}
}
