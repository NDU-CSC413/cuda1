
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "helper_cuda.h"
#include <iostream>

__global__ void kernel(int* a) {
	*a = 17;
	
}

int main() {
	int a = 3;
	int* da = 0;
	cudaMalloc(&da, sizeof(int));
	kernel << <1, 1 >> > (da);
	cudaMemcpy(&a, da, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << a << '\n';
	cudaFree(da);

}