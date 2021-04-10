#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// https://developer.nvidia.com/blog/six-ways-saxpy/

__global__ void saxpy(int n, float a, float*  x, float*  y,float *z) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)z[i] = a * x[i] + y[i];
}

__global__ void kernel(float* a, float* b, float* c) {
	int idx = threadIdx.x;
	c[idx] = a[idx] + b[idx];
}

int main() {
	int N = 1024;
	float* a, * b, * c;
	float* da, * db, * dc;
	a = (float*)malloc(N * sizeof(float));
	b = (float*)malloc(N * sizeof(float));
	c = (float*)malloc(N * sizeof(float));

	cudaMalloc(&da, N * sizeof(float));
	cudaMalloc(&db, N * sizeof(float));
	cudaMalloc(&dc, N * sizeof(float));
	for (int i = 0; i < N; ++i) {
		a[i] = i;
		b[i] = 2 * i;
	}
	cudaMemcpy(da, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, N * sizeof(float), cudaMemcpyHostToDevice);

	kernel << <1, N >> > (da, db, dc);
	cudaMemcpy(c, dc, N * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; ++i)
		std::cout << c[i] << ' ';
	std::cout << std::endl;
	saxpy << <4, 256 >> > (N, 3.5, da, db, dc);
	cudaMemcpy(c, dc, N * sizeof(float), cudaMemcpyDeviceToHost);

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