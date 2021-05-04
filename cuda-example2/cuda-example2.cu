#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 *  kernel()->__global__ void
 * add two arrays in parallel
 * @param a 
 * @param b
 * @param c result of addition
 * @return 
 */

__global__ void kernel(float* a, float* b, float* c) {
	int idx = threadIdx.x;
	c[idx] = a[idx] + b[idx];
}

int main() {
	/* maximum threads per block */
	const int n = 1024;
	float* a, * b, * c;
	float* da, * db, * dc;
	a = (float*)malloc(n * sizeof(float));
	b = (float*)malloc(n * sizeof(float));
	c = (float*)malloc(n * sizeof(float));

	cudaMalloc(&da, n * sizeof(float));
	cudaMalloc(&db, n * sizeof(float));
	cudaMalloc(&dc, n * sizeof(float));
	for (int i = 0; i < n; ++i) {
		a[i] = i;
		b[i] = 2 * i;
	}
	cudaMemcpy(da, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice);

	kernel << <1, n >> > (da, db, dc);
	cudaMemcpy(c, dc, n * sizeof(float), cudaMemcpyDeviceToHost);
	

	for (int i = 0; i < 10; ++i)
		std::cout << c[i] << ' ';
	std::cout << std::endl;
	free(a);
	free(b);
	free(c);
	cudaFree(db);
	cudaFree(dc);
	cudaFree(da);

}