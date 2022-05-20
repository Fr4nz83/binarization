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
	uint32_t *arr = new uint32_t[dataset.size() * H * W];
	return(arr);
}
