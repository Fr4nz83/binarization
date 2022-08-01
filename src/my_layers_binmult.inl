#include "math_constants.h"


// *** FORWARD DECLARATIONS OF KERNELS USED BY THIS CLASS *** //

__global__ void PackWeight32(const float* __restrict__ A, unsigned* B,
							 const int A_height, const int A_width);
__global__ void Input_Binarization(BinaryMultiplicationLayer *p);
__global__ void Mat_BinMul(BinaryMultiplicationLayer* p);
__global__ void Mat_BinMul_T(BinaryMultiplicationLayer* p);
__global__ void Mat_BinMul_OutBin(BinaryMultiplicationLayer* p);
__global__ void Mat_BinMul_T_OutBin(BinaryMultiplicationLayer* p);



// *** CTORS/DTOR DEFINITIONS *** //

BinaryMultiplicationLayer::BinaryMultiplicationLayer(const char* name,
													 const unsigned& weigths_height, // This corresponds to number of features.
													 const unsigned& weigths_width,  // This corresponds to the number of hidden units.
													 const float* weights,
													 const float* bias,
													 const bool& binarized_input,
													 const bool& binarize_output,
													 const bool& transpose_output,
													 const bool& apply_gelu) :
input_gpu(NULL),
input_bin_gpu(NULL),
input_height(0),
input_width(weigths_height),
weights_gpu(NULL),
weights_height(weigths_height),
weights_width(weigths_width),
bias_gpu(NULL),
output_gpu(NULL),
output_bin_gpu(NULL),
gpu(NULL),
binarized_input(binarized_input),
binarize_output(binarize_output),
transpose_output(transpose_output),
apply_gelu(apply_gelu)
{
	strncpy(this->name, name, 8);
	std::cout << "Invoking constructor for " << this->name << std::endl;

	// Load and binarize the weights on GPU.
	this->init_bin_weights(weights, bias);
}



// *** PUBLIC METHODS DEFINITIONS *** //

void BinaryMultiplicationLayer::release()
{
	std::cout << this->name << ": dealloc CUDA resources..." << std::endl;


	this->input_height = 0;


	// Dealloc data space (may be NULL in case this layer is connected to other layers).
	if(this->input_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->input_gpu) );
		this->input_gpu = NULL;
	}

	if(this->input_bin_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->input_bin_gpu) );
		this->input_bin_gpu = NULL;
	}

	if(this->weights_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->weights_gpu) );
		this->weights_gpu = NULL;
	}

	if(this->bias_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->bias_gpu) );
		this->bias_gpu = NULL;
	}

	if(this->output_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->output_gpu) );
		this->output_gpu = NULL;
	}

	if(this->output_bin_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->output_bin_gpu) );
		this->output_bin_gpu = NULL;
	}
}

void BinaryMultiplicationLayer::init_bin_weights(const float* weights, const float* bias)
{
	float* tmp_fp_weights_gpu;

	// Copy the float weights from CPU to GPU.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(tmp_fp_weights_gpu), this->weight_bytes()));
	CUDA_SAFE_CALL(cudaMemcpy(tmp_fp_weights_gpu, weights, this->weight_bytes(), cudaMemcpyHostToDevice));


	// Allocate the memory required by the binarized weights.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->weights_gpu), this->weight_bit_bytes()));
	std::cout << "Memory required by the binarized weights: " << this->weight_bit_bytes() << " bytes." << std::endl;


	// Binarize the float weights.
	// NOTA: con dim3 le dimensioni non specificate vengono lasciate pari a 1.
	// NOTA2: qui sotto il kernel e' lanciato con una griglia di blocchi di thread con dimensione pari a "ceil(height/32) x ceil(width/32) x 1".
	// NOTA3: il kernel richiede che ogni thread-block giri con soli 32 thread (quindi un solo warp per thread-block).
	PackWeight32 <<<dim3(CEIL(weights_height), CEIL(weights_width)), 32>>>
				 	(tmp_fp_weights_gpu, this->weights_gpu, this->weights_height, this->weights_width);


	// Release the device memory temporarily used to binarize the float weights.
	CUDA_SAFE_CALL(cudaFree(tmp_fp_weights_gpu));


	// Copy the bias info to the GPU.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->bias_gpu), this->weights_width * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(this->bias_gpu, bias, this->weights_width * sizeof(float), cudaMemcpyHostToDevice));
}

BinaryMultiplicationLayer* BinaryMultiplicationLayer::ready()
{
	// Various sanity checks before executing the layer.
	if(this->input_bin_gpu == NULL || this->input_size() == 0)
	{
		std::cout << "ERROR: Memory for binarized input has not been allocated!" << std::endl;
		exit(1);
	}

	if(this->binarize_output)
	{
		if(this->output_bin_gpu == NULL || this->output_bit_bytes() == 0)
		{
			std::cout << "ERROR: Space for binarized output has not been allocated/initialized on the GPU." << std::endl;
			exit(1);
		}
	}
	else
	{
		if(this->output_gpu == NULL || this->output_bytes() == 0)
		{
			std::cout << "ERROR: Output has not been allocated/initialized on the GPU." << std::endl;
			exit(1);
		}
	}

	if(this->weights_gpu == NULL)
	{
		std::cout << "ERROR: Weights have not been copied to the GPU." << std::endl;
		exit(1);
	}

	if(this->bias_gpu == NULL)
	{
		std::cout << "ERROR: Biases have not been copied to the GPU." << std::endl;
		exit(1);
	}


	// Allocate shadow copy of this instance on GPU.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->gpu), sizeof(BinaryMultiplicationLayer)));
	CUDA_SAFE_CALL(cudaMemcpy(this->gpu, this, sizeof(BinaryMultiplicationLayer), cudaMemcpyHostToDevice));


	// Return the pointer to the shadow copy (to be used within a kernel).
	return this->gpu;
}

void BinaryMultiplicationLayer::allocate_output_gpu()
{
	// Allocate space for output.
	if(this->binarize_output == false)
	{
		CUDA_SAFE_CALL(cudaMalloc((void**)&(this->output_gpu), this->output_bytes()));
	}
	else
	{
		CUDA_SAFE_CALL(cudaMalloc((void**)&(this->output_bin_gpu), this->output_bit_bytes()));
	}
}

void BinaryMultiplicationLayer::load_input_gpu(void* input, unsigned input_height)
{
	this->input_height = input_height;


	// Check if we need to reset the state of the binarized input.
	if(this->input_bin_gpu != NULL)
		CUDA_SAFE_CALL(cudaFree(this->input_bin_gpu));


	// Allocate the memory required by the binarized input.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->input_bin_gpu), this->input_bit_bytes()));
	std::cout << "Memory required by the binarized input: " << this->input_bit_bytes() << " bytes." << std::endl;


	// Check if we are dealing with input that has already been binarized or not.
	if(this->binarized_input == false)
	{
		if(this->input_gpu != NULL)
			CUDA_SAFE_CALL(cudaFree(this->input_gpu));

		// Allocate the memory required by the FP input, and then copy the input data from CPU to GPU.
		CUDA_SAFE_CALL(cudaMalloc((void**)&(this->input_gpu), this->input_bytes()));
		CUDA_SAFE_CALL(cudaMemcpy(this->input_gpu, (float*)input, this->input_bytes(), cudaMemcpyHostToDevice));
		std::cout << "Memory required by the FP input: " << this->input_bytes() << " bytes." << std::endl;

		// Set the initial state of the binarized input.
		CUDA_SAFE_CALL(cudaMemset(this->input_bin_gpu, 0, this->input_bit_bytes()));
	}
	else
	{
		CUDA_SAFE_CALL(cudaMemcpy(this->input_bin_gpu, (unsigned*)input, this->input_bit_bytes(), cudaMemcpyHostToDevice));
	}


	// Allocate the output of the GPU.
	this->allocate_output_gpu();
}

inline void BinaryMultiplicationLayer::set_input_gpu(void* input_gpu, unsigned input_height)
{
	this->input_height = input_height;


	// Check if we need to reset the state of the binarized input.
	if(this->input_bin_gpu != NULL)
		CUDA_SAFE_CALL(cudaFree(this->input_bin_gpu));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->input_bin_gpu), this->input_bit_bytes()));


	if(this->binarized_input == false)
	{
		if(this->input_gpu != NULL)
			CUDA_SAFE_CALL(cudaFree(this->input_gpu));

		this->input_gpu = (float*)input_gpu;

		CUDA_SAFE_CALL(cudaMemset(this->input_bin_gpu, 0, this->input_bit_bytes()));
	}
	else
	{
		this->input_bin_gpu = (unsigned*)input_gpu;
	}

	// Allocate the output of the GPU.
	this->allocate_output_gpu();
}

void BinaryMultiplicationLayer::download_output_gpu(void* output)
{
	if(this->binarize_output == false)
	{
		CUDA_SAFE_CALL(cudaMemcpy(output, this->output_gpu, this->output_bytes(), cudaMemcpyDeviceToHost));
	}
	else
	{
		CUDA_SAFE_CALL(cudaMemcpy(output, this->output_bin_gpu, this->output_bit_bytes(), cudaMemcpyDeviceToHost));
	}
}

void BinaryMultiplicationLayer::execute_layer()
{
	cudaEvent_t start_bin, end_bin, end_mult;
	cudaEventCreate(&start_bin);
	cudaEventCreate(&end_bin);
	cudaEventCreate(&end_mult);


	// 1 - Prepare the layer for execution.
	BinaryMultiplicationLayer* gpu_copy = this->ready();


	// 2 - Input binarization kernel execution.
	cudaEventRecord(start_bin);
	if(!this->binarized_input)
	{
		std::cout << "Binarizing input..." << std::endl;
		Input_Binarization <<<1000, 128>>> (gpu_copy);
	}
	cudaEventRecord(end_bin);


	// 3 - Input binarization kernel execution.
	std::cout << "Binary matrix multiplication..." << std::endl;
	if(!this->transpose_output)
	{
		if(!this->binarize_output)
			Mat_BinMul <<<10000, 32>>> (gpu_copy);
		else
			Mat_BinMul_OutBin <<<10000, 32>>> (gpu_copy);
	}
	else
	{
		if(!this->binarize_output)
			Mat_BinMul_T <<<10000, 32>>> (gpu_copy);
		else
			Mat_BinMul_T_OutBin <<<10000, 32>>> (gpu_copy);
	}
	cudaEventRecord(end_mult);
	cudaEventSynchronize(end_mult);


	float ms_bin, ms_mult;
	cudaEventElapsedTime(&ms_bin, start_bin, end_bin);
	cudaEventElapsedTime(&ms_mult, end_bin, end_mult);
	std::cout << "Input binarization time: " << ms_bin << " ms." << std::endl;
	std::cout << "Bin mult time: " << ms_mult << " ms." << std::endl;


	cudaEventDestroy(start_bin);
	cudaEventDestroy(end_bin);
	cudaEventDestroy(end_mult);
}



// *** CUDA KERNELS *** //

/** @brief Binarize and pack weight matrix into 32-bit unsigned matrix.
 *
 *  Binarization function to convert a row-major 32-bit floating-point weight matrix into
 *  into a binarized weight matrix where blocks of 32 32-bit-columns are arranged into a row-major format.
 *  This is for the preparation of the weight matrices for the binary matrix multiplication.
 *
 *  @return Void.
 *
 *  @note the keyword restricts essentially indicates that A is a read-only variable, thus favoring caching.
 */
__global__ void PackWeight32(const float* __restrict__ A, unsigned* B,
							 const int A_height, const int A_width)
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const unsigned bx = blockIdx.x; // Indice associato al "# righe / 32" (i.e., "height / 32") della matrice dei pesi.
    const unsigned by = blockIdx.y; // Indice associato al "# colonne / 32" (i.e., "width / 32") della matrice dei pesi.

    unsigned Bval=0;
    #pragma unroll
    for (int i=0; i < 32; i++)
    {
    	// At each loop the warp is reading a sub-row of 32 values, and every thread is binarizing a sub-column of 32 values.
        float f0 = ((by*32 + laneid < A_width) && (bx*32 + i < A_height)) ?
        		   A[((bx*32+i) * A_width) + by * 32 + laneid] :
				   -1.0f; // NOTE: out-of-bounds values are set to -1 (which corresponds to padding for the upcoming binarization).

        // NOTA: shift to the left does a 180 degree rotation of the bits representing the sub-column of 32 values the thread is accumulating, i.e.,
        // 		 LSB becomes MSB.
        Bval = (Bval << 1) | (f0 >= 0 ? 1 : 0);
    }


    // Now each thread in the warp has to output the 32-bit sub-column it has accumulated previously.
    // The overall layout of the binarized output follows that of the **row-major** bit-packing format, as shown in Sec. 5.1 (Figura 4) of the paper.
    //
    // MORE PRECISELY: we are writing a sequence of stripes of "32 x width" binarized column-major values -- hence the row-major arrangement of the
    // 				   blocks of binarized valjes.
    // 				   Each stripe has a size of "32 x ceil(width/32)" 32 bit-columns. Each stripe is made of "ceil(width/32)" blocks, with each block
    //				   containing 32x32 binarized values.
    // if (laneid < A_height * A_width) // NOTE: this check should be safely removable.
    B[bx*(gridDim.y*32) + by*32 + laneid] = Bval;
}

/** @brief Binarize and pack weight matrix into 32-bit unsigned matrix.
 *
 *  Binarization function to convert a row-major 32-bit floating-point input matrix into
 *  a binarized input matrix where blocks of 32 32-bit-rows are arranged into a column-major format.
 *  This is for the preparation of the input matrix for binary matrix multiplication.
 *
 *  @return Void.
 */
__global__ void Input_Binarization(BinaryMultiplicationLayer *p)
{
    constexpr uint32_t WARP_SIZE = 32;
	const uint8_t warpid = threadIdx.x / WARP_SIZE;
	const uint8_t laneid = threadIdx.x % WARP_SIZE;
    const uint32_t& threads_per_block = blockDim.x;
    const uint32_t warps_per_block = threads_per_block / WARP_SIZE;


    const int gdx = (CEIL(p->input_height));
    const int gdy = (CEIL(p->input_width));
    const uint32_t input_width = p->input_width;
    const uint32_t input_height = p->input_height;


    // NOTA: qui si assume che ogni blocco esegua al suo interno 32 warp (i.e., 1024 thread).
    // Il dataset viene visto come una matrice ove ogni riga e' un'entita' ed una colonna
    // rappresenta una singola feature.
    for (int bid = blockIdx.x * warps_per_block + warpid; bid < gdx*gdy; bid += gridDim.x * warps_per_block)
    {
        unsigned bx = bid / gdy; // "bx" rappresenta un blocco di 32 righe.
        unsigned by = bid % gdy; // "by" rappresenta un blocco di 32 colonne.
        // NOTA: Una singola coppia di valori (bx,by) rappresenta una sottomatrice di 32x32 valori.

        register unsigned val;
        #pragma unroll
        for (int i=0; i<32; i++)
        {
            // Qui ogni warp va a leggersi una sottoriga di 32 valori (se alcune posizioni vanno fuori
        	// dalla matrice il valore viene impostato a -1).
        	// - bx*32+i => rappresenta l'indice della riga considerata.
        	// - by*32+laneid => rappresenta l'indice della colonna considerata.
        	const uint32_t col = by*32 + laneid;
        	const uint32_t row = bx*32 + i;
            float f0 = ((col < (input_width)) && (row < (input_height))) ?
            		   p->input_gpu[row * (input_width) + col] :
					   -1.0f;

            // I warp si passano le binarizzazioni dei valori fra di loro tramite "ballot". "brev" viene usato
            // quindi usato per ruotare di 180 la riga di bit (portando cosi' il primo bit del blocco da LSB a MSB).
            unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 >= 0 ? 1 : 0));

            // Il thread dentro un warp con laneid == i ha la responsabilita' (successiva) di salvare la bit-row in memoria.
            // Al momento tale bit-row viene storata in val.
            if (laneid == i) val = r0;
        }

        // Finally every active warp writes out in global memory their block of 32 32-bit-rows.
        // Overall, the blocks are arranged in column-major format.
        // NOTE: the if below should be useless and thus have been commented.
        // if (laneid < (p->input_height) * (p->input_width))
        p->input_bin_gpu[by*gdx*32 + bx*32 + laneid] = val;
    }
}

/**
 * @brief This kernel performs the binary multiplication between a binarized input matrix and a binarized weight matrix,
 * 		  and produces the output matrix in row-major format.
 *
 */
__global__ void Mat_BinMul(BinaryMultiplicationLayer* p)
{
    constexpr uint32_t WARP_SIZE = 32;
	const uint8_t warpid = threadIdx.x / WARP_SIZE;
	const uint8_t laneid = threadIdx.x % WARP_SIZE;
    const uint32_t& threads_per_block = blockDim.x;
    const uint32_t warps_per_block = threads_per_block / WARP_SIZE;


    // Compute the overall number of bit-blocks per input height and weights width.
    // NOTE: this serves to index and compute the FP output matrix, as well as the binarized input and weight matrices.
    const int gdx = CEIL(p->input_height); // Height of the binarized output matrix.
    const int gdy = CEIL(p->weights_width); // Width of the binarized output matrix.


    // Here every "bid" represents a warp that computes a sub-matrix of 32x32 values in the output matrix.
    for (int bid = blockIdx.x * warps_per_block + warpid; bid < gdx*gdy; bid += gridDim.x * warps_per_block)
    {
    	unsigned bx = bid / gdy; // Block index on the input height dimension.
        unsigned by = bid % gdy; // Block index on the weights width dimension.

        // DEBUG.
        // if(laneid == 0) printf("Processing block (%d,%d)\n", by,bx);


        const unsigned* input_sub = &(p->input_bin_gpu[bx*32]); // RECALL: Input matrix is made of column-major arranged blocks (hence the bx), each
        												    	// containing 32 32-bit-rows.
        const unsigned* weight_sub = &(p->weights_gpu[by*32]);  // RECALL: Weight matrix is made of row-major arranged blocks (hence the by),
        													    // each containing 32 32-bit-columns.


        // Here we perform a warp-level vector-vector binary multiplication using the
        // common dimension between the input and weight matrices.
        register int Cm[32] = {0};
        for (int i=0; (i*32) < (p->input_width); i++)
        {
            unsigned r0 = input_sub[i*32*gdx + laneid]; // Ogni thread legge una 32-bit sub-row
            unsigned r1 = weight_sub[i*32*gdy + laneid];

            #pragma unroll
            for (int j=0; j<32; j++) // Data una bit-row della matrice di input, qui cicliamo sulle bit-columns della matrice dei pesi.
            {
            	// Original version: every thread has in charge a bit-row of the input matrix...
            	// NOTE: this forces to write the full-precision output in column-major format if one wants to use coalescing.
            	//		 This, of course, may be useful if one wants to get the output transposed for free.
                // unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, j); // from lane-j, r1 of weight matrix
                // Cm[j] += __popc(r0 ^ r2);


            	// Alternative approach: every thread has in charge a bit-column of the weight matrix...
                // This allows to write the output matrix in row-major format using coalescing.
            	unsigned r2 = __shfl_sync(0xFFFFFFFF, r0, j); // from lane-j, r0 of input matrix
				Cm[j] += __popc(r1 ^ r2);
            }
        }


        // Compute the final results of the binary multiplication by applying the formula for -1/1 on the "popc(xor)" results that have been accumulated.
		#pragma unroll
        for(uint8_t i = 0; i < 32; i++)
        	Cm[i] = (int)p->input_width - 2 * Cm[i];


        // Now, the threads within a warp must output the sub-matrix of 32x32 values they've built in:
        // full precision and row-major format.
        const uint32_t start_row = bx * 32;
        const uint32_t end_row = min(start_row + 32, p->input_height);
        const uint32_t start_column = by * 32;
        const uint32_t end_column = min(start_column + 32, p->weights_width);


        // Read the biases associated to the interval of the columns of the weight matrix involved
        // by the output block presently considered by this warp.
        float bias = (start_column + laneid < end_column) ? p->bias_gpu[start_column + laneid] : 0;


        // Write out the output block.
        for(uint32_t row = start_row; row < end_row; row++)
        {
            float* output_sub = &(p->output_gpu[row * p->weights_width + start_column]);
        	if(start_column + laneid < end_column)
        	{
        		// DEBUG.
        		// printf("thread %d is writing value %f! R:%d SR:%d ER:%d WW:%d DIFF:%d\n",
        		//		laneid, (float)Cm[row - start_row],
        		//		row, start_row, end_row, p->weights_width, row - start_row);


        		// Read the final result of the binary multiplication.
        		float res = (float)Cm[row - start_row];

        		// Apply the bias.
        		res += bias;

        		// Apply the GELU.
        		res = p->apply_gelu ? (0.5 * res) * (1 + tanhf( sqrtf(2/CUDART_PI_F) * (res + 0.044715 * powf(res, 3)) )) : res;

        		// Write out the final result.
        		output_sub[laneid] = res;
        	}
        }
    }
}

/**
 * @brief This kernel performs the binary multiplication between a binarized input matrix and a boinarized weight matrix,
 * 		  and produces the ***transposed*** output matrix in row-major format.
 *
 */
__global__ void Mat_BinMul_T(BinaryMultiplicationLayer* p)
{
    constexpr uint32_t WARP_SIZE = 32;
	const uint8_t warpid = threadIdx.x / WARP_SIZE;
	const uint8_t laneid = threadIdx.x % WARP_SIZE;
    const uint32_t& threads_per_block = blockDim.x;
    const uint32_t warps_per_block = threads_per_block / WARP_SIZE;


    // Compute the overall number of bit-blocks per input height and weights width.
    // NOTE: this serves to index and compute the FP output matrix, as well as the binarized input and weight matrices.
    const int gdx = CEIL(p->input_height); // Height of the binarized output matrix.
    const int gdy = CEIL(p->weights_width); // Width of the binarized output matrix.


    // Here every "bid" represents a warp that computes a sub-matrix of 32x32 values in the output matrix.
    for (int bid = blockIdx.x * warps_per_block + warpid; bid < gdx*gdy; bid += gridDim.x * warps_per_block)
    {
    	unsigned bx = bid / gdy; // Block index on the input height dimension.
        unsigned by = bid % gdy; // Block index on the weights width dimension.

        // DEBUG.
        // if(laneid == 0) printf("Processing block (%d,%d)\n", by,bx);


        const unsigned* input_sub = &(p->input_bin_gpu[bx*32]); // RECALL: Input matrix is made of column-major arranged blocks (hence the bx), each
        												    	// containing 32 32-bit-rows.
        const unsigned* weight_sub = &(p->weights_gpu[by*32]);  // RECALL: Weight matrix is made of row-major arranged blocks (hence the by),
        													    // each containing 32 32-bit-columns.


        // Here we perform a warp-level vector-vector binary multiplication using the
        // common dimension between the input and weight matrices.
        register int Cm[32] = {0};
        for (int i=0; (i*32) < (p->input_width); i++)
        {
            unsigned r0 = input_sub[i*32*gdx + laneid]; // Ogni thread del warp legge una 32-bit sub-row della input matrix
            unsigned r1 = weight_sub[i*32*gdy + laneid]; // Ogni thread del warp legge una 32-bit sub-column della weight matrix

            #pragma unroll
            for (int j=0; j<32; j++) // Data una bit-row della matrice di input, qui cicliamo sulle bit-columns della matrice dei pesi.
            {
            	// Original version: every thread has in charge a bit-row of the input matrix...
            	// NOTE: this forces to write the full-precision output in column-major format if one wants to use coalescing.
            	//		 This, of course, may be useful if one wants to get the output transposed for free.
                unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, j); // from lane-j, r1 of weight matrix
                Cm[j] += __popc(r0 ^ r2);
            }
        }

        // Compute the final results of the binary multiplication by applying the formula for -1/1 on the "popc(xor)" results that have been accumulated.
		#pragma unroll
        for(uint8_t i = 0; i < 32; i++)
        	Cm[i] = (int)p->input_width - 2 * Cm[i];



        // Now, the threads within a warp must output the sub-matrix of 32x32 values they've built in:
        // full precision and column-major format.
        const uint32_t start_row = bx * 32;
        const uint32_t end_row = min(start_row + 32, p->input_height);
        const uint32_t start_column = by * 32;
        const uint32_t end_column = min(start_column + 32, p->weights_width);

        // Read the biases associated to the interval of the columns of the weight matrix involved
        // by the output block presently considered by this warp. Uses coalescing.
        float bias = (start_column + laneid < end_column) ? p->bias_gpu[start_column + laneid] : 0;

        // Write out the output block in column-major format -- this has the effect of writing the transposed output matrix
        // in row-major format.
        for(uint32_t column = start_column; column < end_column; column++)
        {
            float* output_sub = &(p->output_gpu[column * p->input_height + start_row]); // Compute the output address for the sub-column to write.
            float bias_col = __shfl_sync(0xFFFFFFFF, bias, column - start_column); // Here each thread retrieves the bias to apply to this column
            																	   // from the thread in the warp that has read it before.

            if(start_row + laneid < end_row)
        	{
        		// DEBUG.
        		/*printf("thread %d is writing value %f! R:%d SR:%d ER:%d C:%d SC:%d EC:%d WW:%d DIFFR:%d DIFFC:%d\n",
        				laneid, (float)Cm[column - start_column],
						start_row + laneid, start_row, end_row,
						column, start_column, end_column,
						p->weights_width,
						laneid,
						column - start_column);*/


        		// Read the final result of the binary multiplication.
        		float res = (float)Cm[column - start_column];

        		// Apply the bias associated with the currently considered column.
        		res += bias_col;

        		// Apply the GELU.
        		res = p->apply_gelu ? (0.5 * res) * (1 + tanhf( sqrtf(2/CUDART_PI_F) * (res + 0.044715 * powf(res, 3)) )) : res;

        		// Write out the final result.
        		output_sub[laneid] = res;
        	}
        }
    }
}

/**
 * @brief This kernel performs the binary multiplication between a binarized input matrix and a binarized weight matrix,
 * 		  and produces the output matrix in binarized format, where the blocks are arranged in column-major format and each
 * 		  block contains 32 32-bit-rows.
 */
__global__ void Mat_BinMul_OutBin(BinaryMultiplicationLayer* p)
{
    constexpr uint32_t WARP_SIZE = 32;
	const uint8_t warpid = threadIdx.x / WARP_SIZE;
	const uint8_t laneid = threadIdx.x % WARP_SIZE;
    const uint32_t& threads_per_block = blockDim.x;
    const uint32_t warps_per_block = threads_per_block / WARP_SIZE;


    // Compute the overall number of bit-blocks per input height and weights width.
    // NOTE: this serves to index and compute the FP output matrix, as well as the binarized input and weight matrices.
    const int gdx = CEIL(p->input_height); // Height of the binarized output matrix.
    const int gdy = CEIL(p->weights_width); // Width of the binarized output matrix.


    // Here every "bid" represents a warp that computes a sub-matrix of 32x32 values in the output matrix.
    for (int bid = blockIdx.x * warps_per_block + warpid; bid < gdx*gdy; bid += gridDim.x * warps_per_block)
    {
    	unsigned bx = bid / gdy; // Block index on the input height dimension.
        unsigned by = bid % gdy; // Block index on the weights width dimension.

        // DEBUG.
        // if(laneid == 0) printf("Processing block (%d,%d)\n", by,bx);


        const unsigned* input_sub = &(p->input_bin_gpu[bx*32]); // RECALL: Input matrix is made of column-major arranged blocks (hence the bx), each
        												    	// containing 32 32-bit-rows.
        const unsigned* weight_sub = &(p->weights_gpu[by*32]);  // RECALL: Weight matrix is made of row-major arranged blocks (hence the by),
        													    // each containing 32 32-bit-columns.


        // Here we perform a warp-level vector-vector binary multiplication using the
        // common dimension between the input and weight matrices.
        register int Cm[32] = {0};
        for (int i=0; (i*32) < (p->input_width); i++)
        {
            unsigned r0 = input_sub[i*32*gdx + laneid]; // Ogni thread legge una 32-bit sub-row
            unsigned r1 = weight_sub[i*32*gdy + laneid];

            #pragma unroll
            for (int j = 0; j < WARP_SIZE; j++) // Data una bit-row della matrice di input, qui cicliamo sulle bit-columns della matrice dei pesi.
            {
            	// Original version: every thread has in charge a bit-row of the input matrix...
            	// NOTE: this forces to write the full-precision output in column-major format if one wants to use coalescing.
            	//		 This, of course, may be useful if one wants to get the output transposed for free.
                unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, j); // from lane-j, r1 of weight matrix
                Cm[j] += __popc(r0 ^ r2);


            	// Alternative approach: every thread has in charge a bit-column of the weight matrix...
                // This allows to write the output matrix in row-major format using coalescing.
            	// unsigned r2 = __shfl_sync(0xFFFFFFFF, r0, j); // from lane-j, r0 of input matrix
				// Cm[j] += __popc(r1 ^ r2);
            }
        }


        // Compute the final results of the binary multiplication by applying the formula for -1/1 on the "popc(xor)" results that have been accumulated.
		#pragma unroll
        for(uint8_t i = 0; i < WARP_SIZE; i++)
        	Cm[i] = (int)p->input_width - 2 * Cm[i];


        // Now, the threads within a warp must output the sub-matrix of 32x32 values they've built in:
        // full precision and row-major format.
        const uint32_t start_row = bx * 32;
        const uint32_t end_row = min(start_row + 32, p->input_height);
        const uint32_t start_column = by * 32;
        const uint32_t end_column = min(start_column + 32, p->weights_width);


        // Read the biases associated to the interval of the columns of the weight matrix involved
        // by the output block presently considered by this warp.
        float bias = (start_column + laneid < end_column) ? p->bias_gpu[start_column + laneid] : 0;


        // Write out the binarized output block that has been assigned to this warp.
        unsigned* output_sub = &(p->output_bin_gpu[by*(gdx*32) + bx*32]);
        unsigned val = 0;
        const uint32_t row = start_row + laneid;

		// NOTE: The -1 corresponds to padding possibly required for out-of-bound elements.
		// NOTE: The checks in the if allow to pad the out-of-bound elements by letting "res" set to -1.
		#pragma unroll
		for(uint32_t col = start_column; col < start_column + WARP_SIZE; col++)
		{
			float res = -1.;
			float bias_col = __shfl_sync(0xFFFFFFFF, bias, col - start_column); // Here each thread retrieves the bias to apply to this column
			            													    // from the thread in the warp that has read it before.
			if((row < end_row) && (col < end_column))
			{
				// DEBUG.
				// printf("thread %d is writing value %f! R:%d SR:%d ER:%d WW:%d DIFF:%d\n",
				//		laneid, (float)Cm[row - start_row],
				//		row, start_row, end_row, p->weights_width, row - start_row);


				// Read the final result of the binary multiplication.
				res = (float)Cm[col - start_column];

				// Apply the bias.
				res += bias_col;

				// Apply the GELU.
				res = p->apply_gelu ? (0.5 * res) * (1 + tanhf( sqrtf(2/CUDART_PI_F) * (res + 0.044715 * powf(res, 3)) )) : res;
			}

			val = (val << 1) | (res >= 0);
		}

    	// Write out the block of 32 32-bit-rows binarized by this warp (we use coalescing!).
		output_sub[laneid] = val;
    }
}

/**
 * @brief This kernel performs the binary multiplication between a binarized input matrix and a binarized weight matrix,
 * 		  and produces the binarized ***transposed*** output matrix.
 * 		  The binarized output matrix has its blocks arranged in column-major format, and each block contains 32 32-bit-rows.
 */
__global__ void Mat_BinMul_T_OutBin(BinaryMultiplicationLayer* p)
{
    constexpr uint32_t WARP_SIZE = 32;
	const uint8_t warpid = threadIdx.x / WARP_SIZE;
	const uint8_t laneid = threadIdx.x % WARP_SIZE;
    const uint32_t& threads_per_block = blockDim.x;
    const uint32_t warps_per_block = threads_per_block / WARP_SIZE;


    // Compute the overall number of bit-blocks per input height and weights width.
    // NOTE: this serves to index and compute the FP output matrix, as well as the binarized input and weight matrices.
    const int gdx = CEIL(p->input_height); // Height of the binarized output matrix.
    const int gdy = CEIL(p->weights_width); // Width of the binarized output matrix.


    // Here every "bid" represents a warp that computes a sub-matrix of 32x32 values in the output matrix.
    for (int bid = blockIdx.x * warps_per_block + warpid; bid < gdx*gdy; bid += gridDim.x * warps_per_block)
    {
    	unsigned bx = bid / gdy; // Block index on the input height dimension.
        unsigned by = bid % gdy; // Block index on the weights width dimension.

        // DEBUG.
        // if(laneid == 0) printf("Processing block (%d,%d)\n", by,bx);


        const unsigned* input_sub = &(p->input_bin_gpu[bx*32]); // RECALL: Input matrix is made of column-major arranged blocks (hence the bx), each
        												    	// containing 32 32-bit-rows.
        const unsigned* weight_sub = &(p->weights_gpu[by*32]);  // RECALL: Weight matrix is made of row-major arranged blocks (hence the by),
        													    // each containing 32 32-bit-columns.


        // Here we perform a warp-level vector-vector binary multiplication using the
        // common dimension between the input and weight matrices.
        register int Cm[32] = {0};
        for (int i=0; (i*32) < (p->input_width); i++)
        {
            unsigned r0 = input_sub[i*32*gdx + laneid]; // Ogni thread del warp legge una 32-bit sub-row della input matrix
            unsigned r1 = weight_sub[i*32*gdy + laneid]; // Ogni thread del warp legge una 32-bit sub-column della weight matrix

            #pragma unroll
            for (int j=0; j<32; j++) // Data una bit-row della matrice di input, qui cicliamo sulle bit-columns della matrice dei pesi.
            {
            	// Original version: every thread has in charge a bit-row of the input matrix...
				// NOTE: this forces to write the full-precision output in column-major format if one wants to use coalescing.
				//		 This, of course, may be useful if one wants to get the output transposed for free.
				// unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, j); // from lane-j, r1 of weight matrix
				// Cm[j] += __popc(r0 ^ r2);


				// Alternative approach: every thread has in charge a bit-column of the weight matrix...
				// This allows to write the output matrix in row-major format using coalescing.
				unsigned r2 = __shfl_sync(0xFFFFFFFF, r0, j); // from lane-j, r0 of input matrix
				Cm[j] += __popc(r1 ^ r2);
            }
        }

        // Compute the final results of the binary multiplication by applying the formula for -1/1 on the "popc(xor)" results that have been accumulated.
		#pragma unroll
        for(uint8_t i = 0; i < 32; i++)
        	Cm[i] = (int)p->input_width - 2 * Cm[i];



        // Now, the threads within a warp must output the sub-matrix of 32x32 values they've built in:
        // full precision and column-major format.
        const uint32_t start_row = bx * 32;
        const uint32_t end_row = min(start_row + 32, p->input_height);
        const uint32_t start_column = by * 32;
        const uint32_t end_column = min(start_column + 32, p->weights_width);



        unsigned* output_sub = &(p->output_bin_gpu[bx*(gdy*32) + by*32]); // Compute the output address of the binarized sub-column to write.
        unsigned val = 0;
        const uint32_t column = start_column + laneid;

        // Read the biases associated to the interval of the columns of the weight matrix involved
		// by the output block presently considered by this warp. Uses coalescing.
        float bias_col = (column < end_column) ? p->bias_gpu[column] : 0;

		#pragma unroll
        for(uint32_t row = start_row; row < start_row + 32; row++)
        {

            // NOTE: "res" is set to -1 in case of padding required for out-of-bounds elements.
            float res = -1.;
            if((row < end_row) && (column < end_column))
        	{
        		// DEBUG.
        		/*printf("thread %d is writing value %f! R:%d SR:%d ER:%d C:%d SC:%d EC:%d WW:%d DIFFR:%d DIFFC:%d\n",
        				laneid, (float)Cm[column - start_column],
						start_row + laneid, start_row, end_row,
						column, start_column, end_column,
						p->weights_width,
						laneid,
						column - start_column);*/


        		// Read the final result of the binary multiplication.
        		res = (float)Cm[row - start_row];

        		// Apply the bias associated with the currently considered column.
        		res += bias_col;

        		// Apply the GELU.
        		res = p->apply_gelu ? (0.5 * res) * (1 + tanhf( sqrtf(2/CUDART_PI_F) * (res + 0.044715 * powf(res, 3)) )) : res;
        	}

			// Binarize the result and store it in "val" (i.e., each thread is binarizing its row).
			// Each thread must also ensure that the LSB becomes the MSB in the process (this explains the left shift).
			val = (val << 1) | (res >= 0);
        }

        // Now, each thread writes out the rows of 32 values it had in charge in binarized format,
        // and we use the coalescing in the process.
        output_sub[laneid] = val;
    }
}
