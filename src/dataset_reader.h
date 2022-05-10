#pragma once


// *** INCLUDES *** //

#include <stdint.h>
#include <vector>
#include <string>
#include <fstream>



// *** GENERIC TYPES *** //

/** @brief Struct representing a single cifar10 image, which bytes are arranged according to the format outlined in the website.
 *  @note The struct fields are packed.
 *
 * @param H Height of a single image.
 * @param W Width of a single image.
 *
 */
template<int H, int W>
struct __attribute__((__packed__)) cifar_image
{
	uint8_t label;  // Image label.
	uint8_t R[H*W]; // Red channel.
	uint8_t G[H*W]; // Green channel.
	uint8_t B[H*W]; // Blue channel.
};

template<int H, int W>
struct cifar_image_float
{
	uint8_t label;  // Image label.
	float R[H*W];   // Red channel.
	float G[H*W];   // Green channel.
	float B[H*W];   // Blue channel.
	
	cifar_image_float() {};
	
	cifar_image_float(const cifar_image<H, W>& obj)
	{
		this->label = obj.label;
		for(uint32_t i = 0; i < H*W; i++)
		{
			this->R[i] = obj.R[i];
			this->G[i] = obj.G[i];
			this->B[i] = obj.B[i];
		}
	}
	
	cifar_image_float& operator=(const cifar_image<H, W>& obj)
	{
		this->label = obj.label;
		for(uint32_t i = 0; i < H*W; i++)
		{
			this->R[i] = obj.R[i];
			this->G[i] = obj.G[i];
			this->B[i] = obj.B[i];
		}
		
		return *this;
	}
};



// *** CLASSES *** //

template<int H, int W>
class ImgDatasetReader
{
public:

	// *** PUBLIC TYPEDEFS *** //

	typedef cifar_image<H, W> cif_images_t;
	


protected:

	// *** PROTECTED CTORS *** //
	
	ImgDatasetReader(); // Dummy default ctor.



public:

	// *** PUBLIC TYPEDEFS *** //
	
	static std::vector<cifar_image_float<H,W>> read_dataset_cifar10(const std::string filename);
};



// Pull in the templatized functions.
// #include "dataset_reader.inl"
template <int H, int W>
std::vector<cifar_image_float<H,W>> ImgDatasetReader<H,W>::read_dataset_cifar10(const std::string filename)
{
    typedef typename ImgDatasetReader<H,W>::cif_images_t ImageType;
    typedef cifar_image_float<H,W> ImageTypeFloat;
    
    
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
