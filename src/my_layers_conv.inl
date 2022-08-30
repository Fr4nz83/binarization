/*
 * This file contains the definitions of the methods of the class ConvLayer.
 */


// *** PROTECTED METHODS DEFINITIONS *** //

bool ConvLayer::initialize_cuDNN()
{
	std::cout << this->name << " => Initializing cuDNN!" << std::endl;


	cudnnStatus_t status;


	// Allocate cuDNN handle
	status = cudnnCreate(&this->cudnn);
	if (status != CUDNN_STATUS_SUCCESS) return false;
	std::cout << this->name << " => cuDNN handle OK!" << std::endl;


	// Set the algorithm used to compute the convolution.
	this->convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
	// this->convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	// this->convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;

	// Allocate the convolution descriptor.
	cudnnCreateConvolutionDescriptor(&this->convolution_descriptor);
	status = cudnnSetConvolution2dDescriptor(this->convolution_descriptor,
											   /*pad_height=*/this->pad_h,
											   /*pad_width=*/this->pad_w,
											   /*vertical_stride=*/this->stride_vertical,
											   /*horizontal_stride=*/this->stride_horizontal,
											   /*dilation_height=*/1,
											   /*dilation_width=*/1,
											   /*mode=*/CUDNN_CROSS_CORRELATION,
											   /*computeType=*/CUDNN_DATA_FLOAT);
	if (status != CUDNN_STATUS_SUCCESS) return false;
	std::cout << this->name << " => cuDNN convolution descriptor allocation OK!" << std::endl;


	return true;
}

/**
 * @brief This method loads the filter and batch normalization (if required) data needed by the layer.
 *
 * @param filters Pointer to a region of memory containing the values of the filters, according to the NCHW format.
 * @param bn Pointer to a region of memory containing the batch normalization data (one value per output channel).
 */
bool ConvLayer::initialize_filters(const float* filters)
{
	// Read and allocate filters data.
	std::cout << this->name << " => Copying filter data from CPU to GPU..." << std::endl;
	std::cout << this->name << " => Filter bytes: " << filter_bytes() << std::endl;
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->filter_gpu), filter_bytes()));
	CUDA_SAFE_CALL(cudaMemcpy(this->filter_gpu, filters, filter_bytes(), cudaMemcpyHostToDevice));



	cudnnStatus_t status;

	// Allocate kernel tensor data structures.
	cudnnCreateFilterDescriptor(&this->kernel_descriptor);
	status = cudnnSetFilter4dDescriptor(this->kernel_descriptor,
										/*dataType=*/CUDNN_DATA_FLOAT,
										/*format=*/CUDNN_TENSOR_NCHW,
										/*out_channels=*/this->output_channels,
										/*in_channels=*/this->input_channels,
										/*kernel_height=*/this->filter_height,
										/*kernel_width=*/this->filter_width);
	if (status != CUDNN_STATUS_SUCCESS) return false;
	std::cout << this->name << " => cuDNN filter tensor descriptor allocation OK!" << std::endl;


	return true;
}



// *** PUBLIC CTORS DEFINITIONS *** //

ConvLayer::ConvLayer(const char* name,
					 unsigned input_height,
					 unsigned input_width,
					 unsigned filter_height,
					 unsigned filter_width,
					 unsigned input_channels,
					 unsigned output_channels,
					 const float* filters,
					 unsigned stride_vertical,
					 unsigned stride_horizontal,
					 unsigned pad_h,
					 unsigned pad_w,
					 bool same_conv,
					 bool save_residual)
{
	strncpy(this->name, name, 8);

	this->size_batch = 0;
	this->input_height = input_height;
	this->input_width = input_width;
	this->filter_height = filter_height;
	this->filter_width = filter_width;
	this->input_channels = input_channels;
	this->output_channels = output_channels;
	this->stride_vertical = stride_vertical;
	this->stride_horizontal = stride_horizontal;
	this->same_conv = same_conv;

	this->input_gpu = NULL;
	this->output_gpu = NULL;
	this->filter_gpu = NULL;
	this->save_residual_gpu = NULL;
	this->input_descriptor = NULL;
	this->output_descriptor = NULL;
	this->d_workspace = NULL;


	// Calculate the padding required in case we are performing "same convolution".
	this->pad_h = this->same_conv ?
				  (( ( (input_height+stride_vertical-(input_height%stride_vertical)) / stride_vertical - 1) *
					stride_vertical+filter_height-input_height) >> 1) : pad_h;
	this->pad_w = this->same_conv ?
				  (( ( (input_width+stride_horizontal-(input_width%stride_horizontal)) / stride_horizontal - 1) *
					stride_horizontal+filter_width-input_width) >> 1) : pad_w;


	// Calculate the output size.
	this->output_height = this->same_conv ?
						  this->input_height :
						  (int)((((float)this->input_height +  2*this->pad_h - this->filter_height) / this->stride_vertical) + 1);
	this->output_width = this->same_conv ?
						 this->input_width :
						 (int)((((float)this->input_width +  2*this->pad_w - this->filter_width) / this->stride_horizontal) + 1);


	// A few prints.
	std::cout << this->name << " => Input characteristics: C=" << this->input_channels << " H=" << this->input_height << " W=" << this->input_width << std::endl;
	std::cout << this->name << " => Same convolution? " << (this->same_conv ? "YES" : "NO") << " -- Calculated padding: H=" << this->pad_h << " W=" << this->pad_h << std::endl;
	std::cout << this->name << " => Filter characteristics: N=" << this->output_channels << " C=" << this->input_channels << " H=" << this->filter_height << " W=" << this->filter_width << std::endl;
	std::cout << this->name << " => Output characteristics: C=" << this->output_channels << " H=" << this->output_height << " W=" << this->output_width << std::endl;


	// Initialize cuDNN data structures.
	this->initialize_cuDNN();

	// Initialize filters.
	this->initialize_filters(filters);
}



// *** PUBLIC METHODS DEFINITIONS *** //

/**
 * @brief This method is invoked by the class destructor and deallocates CUDA resources.
 */
void ConvLayer::release()
{
	std::cout << "Releasing CUDA resources..." << std::endl;


	this->size_batch = 0;


	if (this->input_gpu != NULL)
	{
		CUDA_SAFE_CALL(cudaFree(this->input_gpu));
		this->input_gpu = NULL;
	}

	if (this->output_gpu != NULL)
	{
		CUDA_SAFE_CALL(cudaFree(this->output_gpu));
		this->output_gpu = NULL;
	}

	if (this->filter_gpu != NULL)
	{
		CUDA_SAFE_CALL(cudaFree(this->filter_gpu));
		this->filter_gpu = NULL;
	}

	if (this->save_residual_gpu != NULL)
	{
		CUDA_SAFE_CALL(cudaFree(this->save_residual_gpu));
		this->save_residual_gpu = NULL;
	}
	if (this->d_workspace != NULL)
	{
		CUDA_SAFE_CALL(cudaFree(this->d_workspace));
		this->d_workspace = NULL;
	}


	cudnnDestroyTensorDescriptor(this->input_descriptor);
	cudnnDestroyTensorDescriptor(this->output_descriptor);
	cudnnDestroyFilterDescriptor(this->kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(this->convolution_descriptor);

	cudnnDestroy(this->cudnn);
}

/**
 * @brief This method performs a few sanity check operations to ensure that everything is going well.
 */
ConvLayer* ConvLayer::ready()
{
	// Pointers sanity check.
	if (this->input_gpu == NULL || this->input_size() == 0)
	{
		fprintf(stderr, "Input data has not been uploaded to GPU.\n");
		exit(1);
	}
	if (this->output_gpu == NULL || this->output_size() == 0)
	{
		fprintf(stderr, "Output on GPU has not been allocated.\n");
		exit(1);
	}
	if (this->filter_gpu == NULL || this->filter_size() == 0)
	{
		fprintf(stderr, "Output on GPU has not been allocated.\n");
		exit(1);
	}
	if (this->save_residual && this->save_residual_gpu == NULL)
	{
		fprintf(stderr, "Residual for saving on GPU has not been allocated.\n");
		exit(1);
	}



	// *** Compute the workspace required by the selected cuDNN algorithm. *** //
	cudnnStatus_t status;
	this->workspace_bytes = 0;
	status = cudnnGetConvolutionForwardWorkspaceSize(this->cudnn,
													 this->input_descriptor,
													 this->kernel_descriptor,
													 this->convolution_descriptor,
													 this->output_descriptor,
													 this->convolution_algorithm,
													 &this->workspace_bytes);
	if (status != CUDNN_STATUS_SUCCESS) return 0;
	std::cout << this->name << " => cuDNN workspace size calc OK!" << std::endl;
	std::cout << this->name << " => Workspace size: " << this->workspace_bytes << " bytes" << std::endl;
	cudaMalloc((void**)&this->d_workspace, this->workspace_bytes);


	return this;
}

void ConvLayer::download_output_gpu(void* output)
{
	CUDA_SAFE_CALL(cudaMemcpy(output, this->output_gpu, this->output_bytes(), cudaMemcpyDeviceToHost));
}

void ConvLayer::download_residual_gpu(float* residual)
{
	CUDA_SAFE_CALL(cudaMemcpy(residual, this->save_residual_gpu, this->output_bytes(), cudaMemcpyDeviceToHost));
}

void ConvLayer::load_input_gpu(const unsigned& size_batch, const std::vector<void*>& input)
{
	// Save the number of images in the dataset.
	this->size_batch = size_batch;


	// Allocate and then set input array on GPU.
	std::cout << this->name << " => Allocating space for, and then loading image data on GPU..." << std::endl;
	std::cout << this->name << " => Number of bytes to be loaded: " << this->input_bytes() << std::endl;
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->input_gpu), this->input_bytes()));
	CUDA_SAFE_CALL(cudaMemcpy(this->input_gpu, input[0], this->input_bytes(), cudaMemcpyHostToDevice));



	// *** ALLOCATE cuDNN INPUT AND OUTPUT TENSORS *** //

	cudnnStatus_t status;

	// Reset the state of the input and output tensors descriptors.
	if(this->input_descriptor != NULL) cudnnDestroyTensorDescriptor(this->input_descriptor);


	// Allocate input tensor data structures.
	cudnnCreateTensorDescriptor(&this->input_descriptor);
	status = cudnnSetTensor4dDescriptor(this->input_descriptor,
										/*format=*/CUDNN_TENSOR_NCHW,
										/*dataType=*/CUDNN_DATA_FLOAT,
										/*batch_size=*/this->size_batch,
										/*channels=*/this->input_channels,
										/*image_height=*/this->input_height,
										/*image_width=*/this->input_width);
	if (status != CUDNN_STATUS_SUCCESS) exit(1);
	std::cout << this->name << " => cuDNN input tensor descriptor allocation OK!" << std::endl;


	// Allocate space for the output.
	this->allocate_output_gpu();
}

void ConvLayer::set_input_gpu(const unsigned& size_batch, const std::vector<void*>& input_gpu)
{
	// Save the number of images in the dataset.
	this->size_batch = size_batch;
	this->input_gpu = static_cast<float*>(input_gpu[0]);



	// *** ALLOCATE cuDNN INPUT AND OUTPUT TENSORS *** //

	cudnnStatus_t status;

	// Reset the state of the input and output tensors descriptors.
	if(this->input_descriptor != NULL) cudnnDestroyTensorDescriptor(this->input_descriptor);


	// Allocate input tensor data structures.
	cudnnCreateTensorDescriptor(&this->input_descriptor);
	status = cudnnSetTensor4dDescriptor(this->input_descriptor,
										/*format=*/CUDNN_TENSOR_NCHW,
										/*dataType=*/CUDNN_DATA_FLOAT,
										/*batch_size=*/this->size_batch,
										/*channels=*/this->input_channels,
										/*image_height=*/this->input_height,
										/*image_width=*/this->input_width);
	if (status != CUDNN_STATUS_SUCCESS) return exit(1);
	std::cout << this->name << " => cuDNN input tensor descriptor allocation OK!" << std::endl;


	// Allocate space for the output.
	this->allocate_output_gpu();
}

void ConvLayer::allocate_output_gpu()
{
	// Allocate space for output.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->output_gpu), this->output_bytes()));

	if(this->save_residual)
		CUDA_SAFE_CALL(cudaMalloc((void**)&(this->save_residual_gpu), this->output_bytes()));



	// *** ALLOCATE cuDNN OUTPUT TENSORS *** //

	cudnnStatus_t status;

	// Reset the state of the input and output tensors descriptors.
	if(this->output_descriptor != NULL) cudnnDestroyTensorDescriptor(this->output_descriptor);


	// Allocate output tensor data structures.
	cudnnCreateTensorDescriptor(&this->output_descriptor);
	status = cudnnSetTensor4dDescriptor(this->output_descriptor,
										  /*format=*/CUDNN_TENSOR_NCHW,
										  /*dataType=*/CUDNN_DATA_FLOAT,
										  /*batch_size=*/this->size_batch,
										  /*channels=*/this->output_channels,
										  /*image_height=*/this->output_height,
										  /*image_width=*/this->output_width);
	if (status != CUDNN_STATUS_SUCCESS) exit(1);
	std::cout << this->name << " => cuDNN output tensor descriptor allocation OK!" << std::endl;
}

void ConvLayer::execute_layer()
{
	cudnnStatus_t status;
	constexpr float alpha = 1, beta = 0;


	status = cudnnConvolutionForward(this->cudnn,
									 &alpha,
									 this->input_descriptor,
									 this->input_gpu,
									 this->kernel_descriptor,
									 this->filter_gpu,
									 this->convolution_descriptor,
									 this->convolution_algorithm,
									 this->d_workspace,
									 this->workspace_bytes,
									 &beta,
									 this->output_descriptor,
									 this->output_gpu);
	if (status != CUDNN_STATUS_SUCCESS)
	{
		std::cout << this->name << " => cuDNN convolution computation KO! ERROR!!" << std::endl;
		exit(1);
	}
	std::cout << this->name << " => cuDNN convolution computation OK!" << std::endl;


	// If we have skip connections, save the output in the appropriate buffer for residuals.
	if(this->save_residual)
		CUDA_SAFE_CALL(cudaMemcpy(this->save_residual_gpu, this->output_gpu, this->output_bytes(), cudaMemcpyDeviceToDevice));
}
