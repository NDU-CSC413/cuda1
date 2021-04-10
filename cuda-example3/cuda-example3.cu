#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void kernel(int* a) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	//a[idx] = blockIdx.x;
	a[idx] = sizeof(int);
}

int main() {
	const int N = 1024;
	const int size = N * sizeof(int);
	int* a;
	int* da = 0;
	a = (int*)malloc(size);
	cudaMalloc(&da, size);

	kernel << <N / 4, 4 >> > (da);
	cudaMemcpy(a, da, size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 30; ++i)
		std::cout << a[i] << ' ';
	std::cout << std::endl;
	cudaFree(da);
	free(a);
}