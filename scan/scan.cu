
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <iostream>
#include <cooperative_groups.h>
using namespace cooperative_groups;


void seq_scan(float* x, float* y, int n) {
	float acc = 0;
	for (int i = 0; i < n; ++i) {
		acc += x[i];
		y[i] = acc;
	}
}
void recursive_scan(float* x, float* y, int n) {
	std::copy(x, x + n, y);
	float* tmp = (float*)malloc(n * sizeof(float));
	std::copy(x, x + n, tmp);
	for (int i = 1; i < n; i *= 2) {
		for (int j = i; j < n; ++j) {
			y[j] += tmp[j - i];
		}
		std::copy(y, y + n, tmp);
	}
	free(tmp);
}
#define BLOCK_SIZE 1024
__global__ void upsweep(float* x, float* y, int size) {

	__shared__ float tmp[BLOCK_SIZE];
	int idx = threadIdx.x;
	tmp[ idx] = x[idx]; tmp[idx + BLOCK_SIZE/2] = x[idx+BLOCK_SIZE/2];
	int offset = 1;
	for (int d= BLOCK_SIZE>>1; d>0; d>>=1) {
		__syncthreads();

		if (idx <d ) {
			tmp[(2 * idx + 2) * offset - 1] += tmp[(2 * idx + 1)*offset - 1];
		}
		offset *= 2;
	}
	/* down sweep */
	if (idx == 0)tmp[BLOCK_SIZE - 1] = 0;

	for (int d = 1; d < BLOCK_SIZE; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (idx < d) {

			int ai = offset * (2 * idx + 1) - 1;     int bi = offset * (2 * idx + 2) - 1;

			float t = tmp[ai]; tmp[ai] = tmp[bi]; tmp[bi] += t;
		}
	}

	 /* since thread idx is not the same as array index this means
	 * threads compute values of different index than the ones they store
	 * we have to make sure all of them have finishied the computation before
	 * we transfer the values back to DRAM
	 */
	__syncthreads();
	y[idx] = tmp[idx]; y[idx + BLOCK_SIZE/2] = tmp[idx + BLOCK_SIZE/2];
}

__global__ void cuda_scan(float* x, float* y, int size) {

	__shared__ float tmp[BLOCK_SIZE];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
		tmp[idx] = x[idx];
	
	__syncthreads();
	float t;
	for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
		if (idx >= stride) {
			__syncthreads();
			t = tmp[idx - stride];
			__syncthreads();
			tmp[idx] += t;

		}
	}
	y[idx] = tmp[idx];
}

int main() {
	const int n = 1<<10;
	const int blockSize =1<<10;
	const int gridSize = n / blockSize;
	float* x, * y, * dx, * dy;
	cudaMalloc(&dx, n * sizeof(float));
	cudaMalloc(&dy, n * sizeof(float));
	x = (float*)malloc(n * sizeof(float));
	y = (float*)malloc(n * sizeof(float));
	memset(y, 0, n * sizeof(float));
	for (int i = 0; i < n; ++i)x[i] = i/7;
	cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);
	
	cuda_scan << <1, n >> > (dx, dy, n);
	upsweep << <1, n / 2 >> > (dx, dy, n);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	const int num_trials = 1000;
	float total1 = 0, time = 0;
	for (int i = 0; i < num_trials; ++i) {
		cudaEventRecord(start, 0);
		cuda_scan << <1,n >> > (dx,dy, n);
		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		total1 += time;
	}
	cudaDeviceSynchronize();
	std::cout << "average duration =" << total1  << "\n";
	float total2 = 0;
	for (int i = 0; i < num_trials; ++i) {
		cudaEventRecord(start, 0);
		upsweep << <1, n/2>> > (dx, dy, n);
		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		total2 += time;
	}
	std::cout << "average duration =" << total2  << "\n";

	cudaMemcpy(y, dy, n * sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < n; ++i)
	//	if (y[i] != i * (i + 1) / 2)std::cout << "error";
	//for( int i=n-1;i>n-3;--i)
	std::cout << y[n-1]+x[n-1] << "\n";
	std::cout << std::endl;
	std::cout << "ratio=" << total1 / total2 << "\n";
}