#pragma once


// *** INCLUDES *** //

#include <stdint.h>
#include <vector>
#include <string>
#include <fstream>



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


/** @brief Struct representing a single cifar10 image which RGB values are represented via floats.
 *
 * @param H Height of a single image.
 * @param W Width of a single image.
 *
 */
template<int H, int W>
struct cifar_image_float
{
	uint8_t label;  // Image label.
	float R[H*W];   // Red channel.
	float G[H*W];   // Green channel.
	float B[H*W];   // Blue channel.
	
	cifar_image_float() {};
	
	// Copy constructor from cifar_image.
	cifar_image_float(const cifar_image<H, W>& obj)
	{
		this->label = obj.label;
		#pragma unroll
		for(uint32_t i = 0; i < H*W; i++)
		{
			this->R[i] = obj.R[i];
			this->G[i] = obj.G[i];
			this->B[i] = obj.B[i];
		}
	}
	
	// Assignment operator from cifar_image.
	cifar_image_float& operator=(const cifar_image<H, W>& obj)
	{
		this->label = obj.label;
		#pragma unroll
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

	typedef cifar_image<H, W> cif_img_t;
	typedef cifar_image_float<H, W> cif_img_float_t;
	


protected:

	// *** PROTECTED CTORS *** //
	
	ImgDatasetReader(); // Dummy default ctor.



public:

	// *** PUBLIC TYPEDEFS *** //
	
	static std::vector<cif_img_t> read_dataset_cifar10(const std::string& filename);
	static std::vector<cif_img_float_t> read_dataset_cifar10_float(const std::string& filename);
	static uint32_t* transform_dataset_nhwc(const std::vector<cif_img_t>& dataset);
};



// Pull in the class templatized methods.
#include "dataset_reader.inl"
