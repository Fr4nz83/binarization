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
 * @brief This method performs a few sanity check operations to ensure that everything is going well.
 */
bool ConvLayer::ready()
{
	// Pointers sanity check.
	if (this->input_gpu == NULL)
	{
		fprintf(stderr, "Input data has not been uploaded to GPU.\n");
		exit(1);
	}
	if (this->output_gpu == NULL)
	{
		fprintf(stderr, "Output on GPU has not been allocated.\n");
		exit(1);
	}
	if (this->filter_gpu == NULL)
	{
		fprintf(stderr, "Output on GPU has not been allocated.\n");
		exit(1);
	}
	if (this->save_residual && this->save_residual_gpu == NULL)
	{
		fprintf(stderr, "Residual for saving on GPU has not been allocated.\n");
		exit(1);
	}

	return true;
}

/**
 * @brief This method is invoked by the class destructor and deallocates CUDA resources.
 */
void ConvLayer::release()
{
	if (this->output_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->output_gpu) );
		this->output_gpu=NULL;
	}
	if (this->filter_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->filter_gpu) );
		this->filter_gpu = NULL;
	}
	if (this->bn_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->bn_gpu) );
		this->bn_gpu = NULL;
	}
	if (this->save_residual && this->save_residual_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->save_residual_gpu) );
		this->save_residual_gpu=NULL;
	}
}

/**
 * @brief This method is invoked by the class destructor and deallocates cuDNN resources.
 */
void ConvLayer::release_cuDNN()
{
	cudnnDestroyTensorDescriptor(this->input_descriptor);
	cudnnDestroyTensorDescriptor(this->output_descriptor);
	cudnnDestroyFilterDescriptor(this->kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(this->convolution_descriptor);

	cudnnDestroy(this->cudnn);
}



// *** PUBLIC CTORS DEFINITIONS *** //

ConvLayer::ConvLayer(const char* name,
					 unsigned input_height,
					 unsigned input_width,
					 unsigned filter_height,
					 unsigned filter_width,
					 unsigned input_channels,
					 unsigned output_channels,
					 unsigned stride_vertical,
					 unsigned stride_horizontal,
					 unsigned pad_h,
					 unsigned pad_w,
					 bool same_conv,
					 bool apply_bn,
					 bool save_residual)
{
	strncpy(this->name, name, 8);

	this->input_height = input_height;
	this->input_width = input_width;
	this->filter_height = filter_height;
	this->filter_width = filter_width;
	this->input_channels = input_channels;
	this->output_channels = output_channels;
	this->stride_vertical = stride_vertical;
	this->stride_horizontal = stride_horizontal;
	this->apply_bn = apply_bn;
	this->same_conv = same_conv;

	this->input_gpu = NULL;
	this->output_gpu = NULL;
	this->filter_gpu = NULL;
	this->save_residual_gpu = NULL;
	this->input_descriptor = NULL;
	this->output_descriptor = NULL;


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
	std::cout << this->name << " => Input size: H=" << this->input_height << " W=" << this->input_width << " C=" << this->input_channels << std::endl;
	std::cout << this->name << " => Same convolution? " << (this->same_conv ? "YES" : "NO") << " -- Calculated padding: H=" << this->pad_h << " W=" << this->pad_h << std::endl;
	std::cout << this->name << " => Filter characterstics: N=" << this->output_channels << " H=" << this->filter_height << " W=" << this->filter_width << std::endl;
	std::cout << this->name << " => Output size: H=" << this->output_height << " W=" << this->output_width << " C=" << this->output_channels << std::endl;


	// Initialize cuDNN data structures.
	this->initialize_cuDNN();
}



// *** PUBLIC METHODS DEFINITIONS *** //

/**
 * @brief This method loads the filter and batch normalization (if required) data needed by the layer.
 *
 * @param filters Pointer to a region of memory containing the values of the filters, according to the NCHW format.
 * @param bn Pointer to a region of memory containing the batch normalization data (one value per output channel).
 */
bool ConvLayer::initialize_filters(const float* filters, const float* bn)
{
	// Read and allocate filters data.
	std::cout << this->name << " => Copying filter data from CPU to GPU..." << std::endl;
	std::cout << this->name << " => Filter bytes: " << filter_bytes() << std::endl;
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->filter_gpu), filter_bytes()));
	CUDA_SAFE_CALL(cudaMemcpy(this->filter_gpu, filters, filter_bytes(), cudaMemcpyHostToDevice));


	// Read and allocate the variables for batch normalization.
	if(this->apply_bn)
	{
		// TODO: considerare se usarla in futuro.
		std::cout << "Batch normalization currently unsupported..." << std::endl;
		exit(1);

		std::cout << this->name << " => Copying from CPU to GPU batch norm data..." << std::endl;
		CUDA_SAFE_CALL(cudaMalloc((void**)&(this->bn_gpu), bn_bytes()));
		CUDA_SAFE_CALL(cudaMemcpy(this->bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice));
	}
	else this->bn_gpu = NULL;



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

bool ConvLayer::load_input(const unsigned& batch_size, const float* img_data)
{
	// Save the number of images in the dataset.
	this->batch = batch_size;


	// Allocate and then set input array on GPU.
	std::cout << this->name << " => Allocating space for, and then loading image data on GPU..." << std::endl;
	std::cout << this->name << " => Number of bytes to be loaded: " << this->input_bytes() << std::endl;
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->input_gpu), this->input_bytes()));
	CUDA_SAFE_CALL(cudaMemcpy(this->input_gpu, img_data, this->input_bytes(), cudaMemcpyHostToDevice));


	// Allocate and then set output gpu.
	std::cout << this->name << " => Allocating on GPU space for output data..." << std::endl;
	std::cout << this->name << " => Number of bytes to be allocated for the output: " << this->output_bytes() << std::endl;
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->output_gpu), this->output_bytes()));


	// If required, allocate residual data structures for skip connections.
	if(this->save_residual)
	{
		std::cout << this->name << " => Allocating on GPU space for residual data..." << std::endl;
		CUDA_SAFE_CALL(cudaMalloc((void**)&(this->save_residual_gpu), this->output_bytes()));
	}



	// *** ALLOCATE cuDNN INPUT AND OUTPUT TENSORS *** //

	cudnnStatus_t status;

	// Reset the state of the input and output tensors descriptors.
	if(this->input_descriptor != NULL) cudnnDestroyTensorDescriptor(this->input_descriptor);
	if(this->output_descriptor != NULL) cudnnDestroyTensorDescriptor(this->output_descriptor);


	// Allocate input tensor data structures.
	cudnnCreateTensorDescriptor(&this->input_descriptor);
	status = cudnnSetTensor4dDescriptor(this->input_descriptor,
										/*format=*/CUDNN_TENSOR_NCHW,
										/*dataType=*/CUDNN_DATA_FLOAT,
										/*batch_size=*/this->batch,
										/*channels=*/this->input_channels,
										/*image_height=*/this->input_height,
										/*image_width=*/this->input_width);
	if (status != CUDNN_STATUS_SUCCESS) return false;
	std::cout << this->name << " => cuDNN input tensor descriptor allocation OK!" << std::endl;


	// Allocate output tensor data structures.
	cudnnCreateTensorDescriptor(&this->output_descriptor);
	status = cudnnSetTensor4dDescriptor(this->output_descriptor,
										  /*format=*/CUDNN_TENSOR_NCHW,
										  /*dataType=*/CUDNN_DATA_FLOAT,
										  /*batch_size=*/this->batch,
										  /*channels=*/this->output_channels,
										  /*image_height=*/this->output_height,
										  /*image_width=*/this->output_width);
	if (status != CUDNN_STATUS_SUCCESS) return false;
	std::cout << this->name << " => cuDNN output tensor descriptor allocation OK!" << std::endl;


	// Compute the workspace required by the selected algorithm.
	this->workspace_bytes = 0;
	status = cudnnGetConvolutionForwardWorkspaceSize(this->cudnn,
													 this->input_descriptor,
													 this->kernel_descriptor,
													 this->convolution_descriptor,
													 this->output_descriptor,
													 this->convolution_algorithm,
													 &this->workspace_bytes);
	if (status != CUDNN_STATUS_SUCCESS) return false;
	std::cout << this->name << " => cuDNN workspace size calc OK!" << std::endl;
	std::cout << this->name << " => Workspace size: " << this->workspace_bytes << " bytes" << std::endl;
	cudaMalloc((void**)&this->d_workspace, this->workspace_bytes);


	// Sanity check on various pointers and data structures to ensure that the layer is ready to be executed.
	return this->ready();
}

bool ConvLayer::execute_layer()
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
	if (status != CUDNN_STATUS_SUCCESS) return false;
	std::cout << this->name << " => cuDNN convolution computation OK!" << std::endl;

	return true;
}
