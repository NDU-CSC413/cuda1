
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>


int main()
{
	int device;

	cudaDeviceProp properties;
	cudaError_t err = cudaSuccess;
	err = cudaGetDevice(&device);
	err = cudaGetDeviceProperties(&properties, device);
	std::cout << "processor count" << properties.multiProcessorCount << std::endl;
	std::cout << "warp size " << properties.warpSize << std::endl;
	std::cout << "name=" << properties.name << std::endl;
	std::cout << "Compute capability " << properties.major << "." << properties.minor << "\n";
	std::cout << "shared Memory/SM " << properties.sharedMemPerMultiprocessor
		<< std::endl;
	//  std::cout<<"max blocks/SM "<<properties.maxBlocksPerMultiProcessor
	 // <<std::endl;
	if (err == cudaSuccess)
		printf("device =%d\n", device);
	else
		printf("error getting deivce\n");
	return 0;
}
