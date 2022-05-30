// *** PUBLIC METHODS DEFINITIONS *** //

template <int H, int W>
std::vector<typename ImgDatasetReader<H,W>::cif_img_t> ImgDatasetReader<H,W>::read_dataset_cifar10(const std::string& filename)
{
    typedef cif_img_t ImageType;
    
    std::vector<ImageType> vec_img;
    ifstream file(filename, ios::binary);
    if(file.is_open())
    {
        std::cout << "Reading raw CIFAR-10 from " << filename << "...\n";
        while(file.peek() != EOF)
	{
		// Cifar10 data stored in <1xlabel><r:1024><g:1024><b:1024>
		ImageType tmp_img;		
		file.read((char*) &tmp_img, sizeof(ImageType));
		vec_img.push_back(tmp_img);
		
		
		// *** DEBUG *** //
		// std::cout << "DEBUG: image props: " << (uint32_t) vec_img.back().label << " -- "
		//           << (uint32_t) vec_img.back().R[120] << ", " << (uint32_t) vec_img.back().G[120] << ", " << (uint32_t) vec_img.back().B[120] << std::endl;
        }        
    }
    else
    {
        std::cout << "Error in reading CIFAR-10 image file " << filename << "\n";
        exit(1);
    }
    
    std::cout << "Lette " << vec_img.size() << " immagini\n";
    return(vec_img);
}

template <int H, int W>
std::vector<typename ImgDatasetReader<H,W>::cif_img_float_t> ImgDatasetReader<H,W>::read_dataset_cifar10_float(const std::string& filename)
{
    typedef cif_img_float_t ImageTypeFloat;
    
    std::vector<ImageTypeFloat> res;
    auto tmp = ImgDatasetReader<H,W>::read_dataset_cifar10(filename);
    
    for(const auto& el : tmp)
    {
    	res.push_back(el);
    	
    	// *** DEBUG *** //
	// std::cout << "DEBUG: image float props: " << (uint32_t) res.back().label << " -- "
	//           << res.back().R[120] << ", " << res.back().G[120] << ", " << res.back().B[120] << std::endl;
    }
    	
    return(res);
}

template <int H, int W>
uint32_t* ImgDatasetReader<H,W>::transform_dataset_nhwc(const std::vector<cif_img_t>& dataset)
{
	uint32_t num_images = dataset.size();
	uint32_t *arr = new uint32_t[num_images * H * W];

	uint8_t* ptr_img = (uint8_t*) arr;
	for(uint32_t i = 0; i < num_images; i++)
	{
		#pragma unroll
		for(uint32_t h = 0; h < H; h++)
		{
			#pragma unroll
			for(uint32_t w = 0; w < W; w++)
			{
				ptr_img[(i * H * W + h * W + w) * 4 + 0] = dataset[i].R[h * W + w];
				ptr_img[(i * H * W + h * W + w) * 4 + 1] = dataset[i].G[h * W + w];
				ptr_img[(i * H * W + h * W + w) * 4 + 2] = dataset[i].B[h * W + w];
			}
		}
	}

	return(arr);
}

template <int H, int W>
float* ImgDatasetReader<H,W>::transform_dataset_nhwc_float(const std::vector<cif_img_float_t>& dataset)
{
	constexpr uint32_t num_channels = 3;
	uint32_t num_images = dataset.size();
	float *arr = new float[num_images * H * W * num_channels];

	for(uint32_t i = 0; i < num_images; i++)
	{
		const uint32_t offset_img = i * H * W * num_channels;

		#pragma unroll
		for(uint32_t c = 0; c < num_channels; c++)
		{
			const uint32_t offset_color = H * W * c;

			const float *ptr_val = dataset[i].R;
			if(c == 1) ptr_val = dataset[i].G;
			if(c == 2) ptr_val = dataset[i].B;

			#pragma unroll
			for(uint32_t h = 0; h < H; h++)
			{
				const uint32_t offset_height = h * W;

				#pragma unroll
				for(uint32_t w = 0; w < W; w++)
					arr[offset_img + offset_color + offset_height + w] = ptr_val[offset_height + w];
			}
		}
	}

	return(arr);
}
