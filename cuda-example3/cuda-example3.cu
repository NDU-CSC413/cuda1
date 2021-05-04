#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
/**
 * Single precision AX+Y (saxpy).
 */
using Duration = std::chrono::duration<double, std::milli>;

#define TIMEIT(dur,...)\
   {\
    auto start = std::chrono::high_resolution_clock::now();\
    __VA_ARGS__\
    auto end = std::chrono::high_resolution_clock::now();\
     dur = std::chrono::duration<double, std::milli>(end - start);\
}

__host__ void mult(int n, float a, float* x, float* y, float* z) {
	for (int i = 0; i < n; ++i) {
		z[i] = a * x[i] + y[i];
	}
}

__global__ void saxpy(int n, float a, float* x, float* y, float* z) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)z[i] = a * x[i] + y[i];
}


int main() {
	const int n= 1<<28;
	const int block_size = 1 << 10;
	float* x,*y,*z;
	float* dx,*dy,*dz;
	float a = 3.3;
	x = (float *)malloc(n*sizeof(float));
	y = (float*)malloc(n * sizeof(float));
	z = (float*)malloc(n * sizeof(float));
	for (int i = 0; i < n; ++i) {
		x[i] = i;
		y[i] = 2 * i;
	}
	cudaMalloc(&dx, n*sizeof(float));
	cudaMalloc(&dy, n * sizeof(float));
	cudaMalloc(&dz, n * sizeof(float));
	cudaMemcpy(dx, x, n, cudaMemcpyHostToDevice);
	cudaMemcpy(dy, y, n, cudaMemcpyHostToDevice);

	cudaEvent_t kernel_start, kernel_end;
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_end);
	cudaEventRecord(kernel_start);
	saxpy<<<n/block_size,block_size>> > (n,a,dx,dy,dz);
	cudaEventRecord(kernel_end);
	cudaEventSynchronize(kernel_end);
	float time;
	/* elapsed time in millisconds*/
	cudaEventElapsedTime(&time, kernel_start, kernel_end);
	cudaMemcpy(z, dz, n * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "GPU duration =" << time << "\n";
	
	Duration d;
	TIMEIT(d,
		mult(n, a, x, y, z);
	)
	std::cout <<"CPU duration = "<< d.count() << "\n";
	cudaFree(dx);
	cudaFree(dy);
	cudaFree(dz);

	free(x);
	free(y);
	free(z);
}