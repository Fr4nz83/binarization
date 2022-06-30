#include "math_constants.h"


// *** FORWARD DECLARATIONS OF KERNELS USED BY THIS CLASS *** //

__global__ void PackWeight32(const float* __restrict__ A, unsigned* B,
							 const int A_height, const int A_width);
__global__ void Input_Binarization(BinaryMultiplicationLayer *p);
__global__ void Mat_BinMul(BinaryMultiplicationLayer* p);



// *** CTORS/DTOR DEFINITIONS *** //

BinaryMultiplicationLayer::BinaryMultiplicationLayer(const char* name,
													 const unsigned& weigths_height, // This corresponds to number of features.
													 const unsigned& weigths_width,  // This corresponds to the number of hidden units.
													 const float* weights,
													 const float* bias) :
input_gpu(NULL),
input_bin_gpu(NULL),
input_height(0),
input_width(weigths_height),
weights_gpu(NULL),
weights_height(weigths_height),
weights_width(weigths_width),
bias_gpu(NULL),
output_gpu(NULL),
gpu(NULL)
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
	// Dealloc data space (may be NULL in case this layer is connected to other layers).
	if(this->input_gpu == NULL || this->input_size() == 0)
	{
		std::cout << "ERROR: Input data has not been allocated/initialized on the GPU." << std::endl;
		exit(1);
	}

	if(this->input_bin_gpu == NULL || this->input_bit_size() == 0)
	{
		std::cout << "ERROR: Memory for binarized input has not been allocated!" << std::endl;
		exit(1);
	}

	if(this->output_gpu == NULL || this->output_size() == 0)
	{
		std::cout << "ERROR: Output has not been allocated/initialized on the GPU." << std::endl;
		exit(1);
	}

	// Dealloc scale factors vector.
	if(this->weights_gpu == NULL)
	{
		std::cout << "ERROR: Weights have not been copied to the GPU." << std::endl;
		exit(1);
	}

	// Dealloc shift factors vector.
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

void BinaryMultiplicationLayer::load_input_gpu(float* input, unsigned input_height)
{
	this->input_height = input_height;

	// Check if we need to reset the state of the input.
	if(this->input_gpu != NULL)
		CUDA_SAFE_CALL(cudaFree(this->input_gpu));

	// Allocate the memory required by the FP input, and then copy the input data from CPU to GPU.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->input_gpu), this->input_bytes()));
	CUDA_SAFE_CALL(cudaMemcpy(this->input_gpu, input, this->input_bytes(), cudaMemcpyHostToDevice));
	std::cout << "Memory required by the FP input: " << this->input_bytes() << " bytes." << std::endl;

	// Check if we need to reset the state of the binarized input.
	if(this->input_bin_gpu != NULL)
		CUDA_SAFE_CALL(cudaFree(this->input_gpu));

	// Allocate the memory required by the binarized input, and then set the initial state of its elements to 0.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->input_bin_gpu), this->input_bit_bytes()));
	CUDA_SAFE_CALL(cudaMemset(this->input_bin_gpu, 0, this->input_bit_bytes()));
	std::cout << "Memory required by the binarized input: " << this->input_bit_bytes() << " bytes." << std::endl;
}

void BinaryMultiplicationLayer::allocate_output_gpu()
{
	// Allocate space for output.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->output_gpu), this->output_bytes()));
}

void BinaryMultiplicationLayer::download_output_gpu(float* output)
{
	CUDA_SAFE_CALL(cudaMemcpy(output, this->output_gpu, this->output_bytes(), cudaMemcpyDeviceToHost));
}

void BinaryMultiplicationLayer::execute()
{
	// 3 - Prepare the layer for execution.
	BinaryMultiplicationLayer* gpu_copy = this->ready();

	// 4 - Input binarization kernel execution.
	// TODO: this kernel currently requires each block to have 1024 threads (32 warps). Remove this constraint!
	std::cout << "Binarizing output..." << std::endl;
	Input_Binarization <<<1000, 1024>>> (gpu_copy);

	// 5 - Input binarization kernel execution.
	std::cout << "Binary matrix multiplication..." << std::endl;
	Mat_BinMul <<<1000, 32>>> (gpu_copy);
	cudaDeviceSynchronize();
}



// *** CUDA KERNELS *** //

/** @brief Binarize and pack weight matrix into 32-bit unsigned matrix.
 *
 *  Binarization function to convert row-major 32-bit floating-point weight matrix into
 *  bit row-major bit-matrix. This is for the preparation of the weight matrices for
 *  FC layers.
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
    	// At each loop the warp is reading a sub-row of 32 values, and every trhead is binarizing a sub-column of 32 values.
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
    if (laneid < A_height * A_width) // TODO: this check should be safely removable.
        B[bx*(gridDim.y*32) + by*32 + laneid] = Bval;
}

/** @brief Binarize and pack weight matrix into 32-bit unsigned matrix.
 *
 *  Binarization function to convert row-major 32-bit floating-point weight matrix into
 *  bit column-major bit-matrix. This is for the preparation of the input matrix for binary matrix multiplication.
 *
 *  @return Void.
 */
__global__ void Input_Binarization(BinaryMultiplicationLayer *p)
{
    GET_LANEID; // Recover warpid and laneid;
    // constexpr uint32_t WARP_SIZE = 32;
    // const uint32_t& threads_per_block = blockDim.x;
    // const uint32_t warps_per_block = threads_per_block / WARP_SIZE;


    const int gdx = (CEIL(p->input_height));
    const int gdy = (CEIL(p->input_width));

    // NOTA: qui si assume che ogni blocco esegua al suo interno 32 warp (i.e., 1024 thread).
    // Il dataset viene visto come una matrice ove ogni riga e' un'entita' ed una colonna
    // rappresenta una singola feature.
    for (int bid = blockIdx.x*32 + warpid; bid < gdx*gdy; bid += gridDim.x*32)
    {
        unsigned bx = bid / gdy; // "bx" rappresenta un blocco di 32 righe.
        unsigned by = bid % gdy; // "by" rappresenta un blocco di 32 colonne.
        // NOTA: Una singola coppia di valori (bx,by) rappresenta una sottomatrice di 32x32 valori.

        unsigned val;
        #pragma unroll
        for (int i=0; i<32; i++)
        {
            // Qui ogni warp va a leggersi una sottoriga di 32 valori (se alcune posizioni vanno fuori
        	// dalla matrice il valore viene impostato a -1).
        	// - bx*32+i => rappresenta l'indice della riga considerata.
        	// - by*32+laneid => rappresenta l'indice della colonna considerata.
            float f0 = ((by*32 + laneid < (p->input_width)) && (bx*32+i < (p->input_height))) ?
            		   p->input_gpu[(bx*32+i)*(p->input_width) + by*32 + laneid] :
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
        // NOTE: gdx => number of ints required to store a whole bit-column.
        if (laneid < (p->input_height) * (p->input_width))
            p->input_bin_gpu[by*gdx*32 + bx*32 + laneid] = val;
    }
}

/**
 * @brief This kernel processes batch-normalizes the channels of a set of images.
 *
 * @note The kernel assumes that the dataset is stored in NCHW format.
 */
// TODO: this kernel will have to be a __device__ function at some point.
__global__ void Mat_BinMul(BinaryMultiplicationLayer* p)
{
    constexpr uint32_t WARP_SIZE = 32;
	const uint8_t warpid = threadIdx.x / WARP_SIZE;
	const uint8_t laneid = threadIdx.x % WARP_SIZE;
    const uint32_t& threads_per_block = blockDim.x;
    const uint32_t warps_per_block = threads_per_block / WARP_SIZE;


    // Compute the overall number of bit-blocks per input height and weights width.
    // NOTE: this serves to index and compute the output matrix, as well as the binarized input and weight matrices.
    const int gdx = CEIL(p->input_height); // Height of the binarized output matrix.
    const int gdy = CEIL(p->weights_width); // Width of the binarized output matrix.


    // Here every "bid" represents a warp that computes a sub-matrix of 32x32 values in the output matrix.
    // NOTE: it seems here that every thread block has 32 warps (1024 threads). This constraint can be removed by computing
    // the number of warps per block, and then substituting the 32 in this for loop.
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
            	//		 This, of course, may be useful if one wants to transpose the output on the fly.
                // unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, j); // from lane-j, r1 of weight matrix
                // Cm[j] += __popc(r0 ^ r2);


            	// Alternative approach: every thread has in charge a bit-column of the weight matrix)...
                // This allows to write the output matrix in row-major format using coalescing.
            	unsigned r2 = __shfl_sync(0xFFFFFFFF, r0, j); // from lane-j, r0 of input matrix
				Cm[j] += __popc(r1 ^ r2);
            }
        }


        // Compute the final results by applying the binary multiplication formula for -1/1 on the "popc(xor)" results that have been accumulated.
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
        		res = (0.5 * res) * (1 + tanhf( sqrtf(2/CUDART_PI_F) * (res + 0.044715 * powf(res, 3)) ));

        		// Write out the final result.
        		output_sub[laneid] = res;
        	}
        }
    }
}
