template <int H, int W>
std::vector<typename ImgDatasetReader<H,W>::cif_img_float_t> ImgDatasetReader<H,W>::read_dataset_cifar10(const std::string& filename)
{
    typedef cif_img_t ImageType;
    typedef cif_img_float_t ImageTypeFloat;
    
    
    std::vector<ImageTypeFloat> vec_img;
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
		std::cout << "DEBUG: image props float: " << (uint32_t) vec_img.back().label << " -- "
		          << vec_img.back().R[120] << ", " << vec_img.back().G[120] << ", " << vec_img.back().B[120] << std::endl;
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
