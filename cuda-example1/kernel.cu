
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void kernel(int* a) {
	*a = 17;

}

int main() {
	int a = 3;
	int* da = 0;
	/* allocate memory on device. Note the passing the address of da*/
	cudaMalloc(&da, sizeof(int));
	/* launch kernel with 1 block, 1 thread per block */
	kernel << <1, 1 >> > (da);
	/* copy from device to host */
	cudaMemcpy(&a, da, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << a << '\n';
	cudaFree(da);

}