
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
    case 2: // Fermi
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 7: // Volta and Turing
        if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 8: // Ampere
        if (devProp.minor == 0) cores = mp * 64;
        else if (devProp.minor == 6) cores = mp * 128;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

int main()
{
	int device;

	cudaDeviceProp properties;
	cudaError_t err = cudaSuccess;
	err = cudaGetDevice(&device);
	err = cudaGetDeviceProperties(&properties, device);
	std::cout << "processor count " << properties.multiProcessorCount << std::endl;
	std::cout << "warp size " << properties.warpSize << std::endl;
	std::cout << "name= " << properties.name << std::endl;
	std::cout << "Compute capability " << properties.major << "." << properties.minor << "\n";
	std::cout << "shared Memory/SM " << properties.sharedMemPerMultiprocessor
		<< std::endl;
    std::cout << "number of cores " << getSPcores(properties)<<"\n";
	//  std::cout<<"max blocks/SM "<<properties.maxBlocksPerMultiProcessor
	 // <<std::endl;
	if (err == cudaSuccess)
		printf("device =%d\n", device);
	else
		printf("error getting deivce\n");
	return 0;
}
