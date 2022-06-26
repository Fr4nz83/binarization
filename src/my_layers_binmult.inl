// *** FORWARD DECLARATIONS OF KERNELS USED BY THIS CLASS *** //

__global__ void PackWeight32(const float* __restrict__ A, unsigned* B,
							 const int A_height, const int A_width);

__global__ void Input_Binarization(BinaryMultiplicationLayer *p);



// *** CTORS/DTOR DEFINITIONS *** //

BinaryMultiplicationLayer::BinaryMultiplicationLayer(const char* name,
													 const unsigned& input_width,
													 const unsigned& input_height,
													 const unsigned& weigths_width,
													 const float* weights) :
size_batch(0),
input_gpu(NULL),
input_bin_gpu(NULL),
input_width(input_width),
input_height(input_height),
output_gpu(NULL),
weights_width(weigths_width),
weights_height(input_width),
weights_gpu(NULL),
gpu(NULL)
{
	strncpy(this->name, name, 8);
	std::cout << "Invoking constructor for " << this->name << std::endl;

	// Load and binarize the weights on GPU.
	this->init_bin_weights(weights);
}



// *** PUBLIC METHODS DEFINITIONS *** //

void BinaryMultiplicationLayer::release()
{
	std::cout << this->name << ": dealloc CUDA resources..." << std::endl;


	this->size_batch = 0;

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

	if(this->output_gpu != NULL)
	{
		CUDA_SAFE_CALL( cudaFree(this->output_gpu) );
		this->output_gpu = NULL;
	}
}

void BinaryMultiplicationLayer::init_bin_weights(const float* weights)
{
	float* tmp_fp_weights_gpu;

	// Copy the float weights from CPU to GPU.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(tmp_fp_weights_gpu), this->weight_bytes()));
	CUDA_SAFE_CALL(cudaMemcpy(tmp_fp_weights_gpu, weights, this->weight_bytes(), cudaMemcpyHostToDevice));


	// Allocate the memory required by the binarized weights.
	CUDA_SAFE_CALL(cudaMalloc((void**)&(this->weights_gpu), this->weight_bit_bytes()));


	// Binarize the float weights.
	// NOTA: con dim3 le dimensioni non specificate vengono lasciate pari a 1.
	// NOTA2: qui sotto il kernel e' lanciato con una griglia di blocchi di thread con dimensione pari a "ceil(height/32) x ceil(width/32) x 1".
	// NOTA3: il kernel richiede che ogni thread-block giri con soli 32 thread (quindi un solo warp per thread-block).
	PackWeight32 <<<dim3(CEIL(weights_height), CEIL(weights_width)), 32>>>
				 	(tmp_fp_weights_gpu, this->weights_gpu, this->weights_height, this->weights_width);


	// Release the device memory temporarily used to binarize the float weights.
	CUDA_SAFE_CALL(cudaFree(tmp_fp_weights_gpu));
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
    if (laneid < A_height * A_width) // TODO: this if can be removed.
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

        // Finally every active warp writes out in global memory their bit-row.
        // gdx => number of ints required to store a whole bit-column.
        if (laneid < (p->input_height) * (p->input_width))
            p->output_gpu[by*gdx*32 + bx*32 + laneid] = val;
    }
}

/**
 * @brief This kernel processes batch-normalizes the channels of a set of images.
 *
 * @note The kernel assumes that the dataset is stored in NCHW format.
 */
// TODO: this kernel will have to be a __device__ function at some point.
__global__ void Mat_BinMul_Layer(BinaryMultiplicationLayer* p)
{
/*	constexpr uint8_t WARPSIZE = 32;

	const uint8_t warp_id = threadIdx.x / WARPSIZE;
	const uint8_t lane_id = threadIdx.x % WARPSIZE;
	const uint32_t warps_block = blockDim.x / WARPSIZE;
	const uint32_t& block_id = blockIdx.x;
	const uint32_t& num_blocks = gridDim.x;
	const uint32_t num_img_per_grid = num_blocks * warps_block;*/
}
