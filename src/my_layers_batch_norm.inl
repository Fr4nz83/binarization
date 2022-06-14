// *** CTORS/DTOR DEFINITIONS *** //

BatchNormFullPrecLayer::BatchNormFullPrecLayer(const unsigned& size_batch,
											   const unsigned& data_width,
											   const unsigned& data_height,
											   const unsigned& data_channels,
											   const float* scale,
											   const float* shift) :
size_batch(size_batch),
data_gpu(NULL),
data_width(data_width),
data_height(data_height),
data_channels(data_channels),
gpu(NULL)
{
	// Load the scaling scalars (1 per channel) into the GPU.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->scale), data_channels * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(this->scale, scale, data_channels * sizeof(float), cudaMemcpyHostToDevice));

	// Load the shift scalars (1 per channel) into the GPU.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->shift), data_channels * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(this->shift, shift, data_channels * sizeof(float), cudaMemcpyHostToDevice));
}



// *** CUDA KERNELS *** //

/**
 * @brief This kernel processes batch-normalizes the channels of a set of images.
 *
 * @note The kernel assumes that the dataset is stored in NCHW format.
 */
__global__ void BNFPLayer(BatchNormFullPrecLayer* p)
{
	constexpr uint8_t WARPSIZE = 32;

	const uint8_t warp_id = threadIdx.x / WARPSIZE;
	const uint8_t lane_id = threadIdx.x % WARPSIZE;
	const uint32_t warps_block = blockDim.x / WARPSIZE;
	const uint32_t& block_id = blockIdx.x;
	const uint32_t& num_blocks = gridDim.x;
	const uint32_t num_img_per_grid = num_blocks * warps_block;


	const uint32_t img_size = p->data_width * p->data_height;
	const uint32_t num_channels = p->data_channels;


	// Process each color dataset-wise.
	for(uint32_t c = 0; c < num_channels; c++)
	{
		// Read the scale and shift factors associated with the channel currently considered.
		const float scale = p->scale[c];
		const float shift = p->shift[c];

		// Each warp processes an image per loop.
		for(uint32_t id_img = block_id * warps_block + warp_id; id_img < p->size_batch; id_img =+ num_img_per_grid)
		{
			// Compute the memory offset where the values associated with a given image and channel block starts.
			const uint32_t offset_img = id_img * (num_channels * img_size) + (c * img_size);

			// The threads in a warp uses coalescing while reading and then updating the values.
			for(uint32_t i = lane_id; i < img_size; i += WARPSIZE)
				p->data_gpu[offset_img + i] = scale * p->data_gpu[offset_img + i] + shift;
		}
	}
}
