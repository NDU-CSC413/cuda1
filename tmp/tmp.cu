
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <iostream>
#include <cooperative_groups.h>

#define NUM_BANKS 4
#define LOG_NUM_BANKS 2

#define CONFLICT_FREE_OFFSET(n) \
   ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

#define BLOCK_SIZE 1024

__global__ void scan1(float* x, float* y) {

	__shared__ float tmp[BLOCK_SIZE];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidx = threadIdx.x;

	int ai = tidx;
	int bi = tidx + (BLOCK_SIZE / 2);

	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	tmp[ai + bankOffsetA] = x[ai];
	tmp[bi + bankOffsetB] = x[bi];


	//tmp[tidx] = x[idx]; tmp[tidx + BLOCK_SIZE / 2] = x[idx + BLOCK_SIZE / 2];
	int offset = 1;
	for (int d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
		__syncthreads();
		int ai = (2 * tidx + 1) * offset - 1;
		int bi = (2 * tidx + 2) * offset - 1;
		ai += CONFLICT_FREE_OFFSET(ai);
		bi += CONFLICT_FREE_OFFSET(bi);
		if (tidx < d) {
			tmp[bi] += tmp[ai];
		}
		offset *= 2;
	}
	/* down sweep */
	if (tidx == 0)tmp[BLOCK_SIZE - 1 + CONFLICT_FREE_OFFSET(BLOCK_SIZE - 1)] = 0;

	for (int d = 1; d < BLOCK_SIZE; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (tidx < d) {

			int ai = offset * (2 * tidx + 1) - 1;     int bi = offset * (2 * tidx + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			float t = tmp[ai]; tmp[ai] = tmp[bi]; tmp[bi] += t;
		}
	}

	/* since thread idx is not the same as array index this means
	* threads compute values of different index than the ones they store
	* we have to make sure all of them have finishied the computation before
	* we transfer the values back to DRAM
	*/
	__syncthreads();
	y[ai] = tmp[ai + bankOffsetA]; y[bi] = tmp[bi + bankOffsetB];
}
__global__ void scan2(float* x, float* y) {

	__shared__ float tmp[BLOCK_SIZE];
	int idx = threadIdx.x;
	tmp[idx] = x[idx]; tmp[idx + BLOCK_SIZE / 2] = x[idx + BLOCK_SIZE / 2];
	int offset = 1;
	for (int d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
		__syncthreads();

		if (idx < d) {
			tmp[(2 * idx + 2) * offset - 1] += tmp[(2 * idx + 1) * offset - 1];
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
	y[idx] = tmp[idx]; y[idx + BLOCK_SIZE / 2] = tmp[idx + BLOCK_SIZE / 2];
}

int main() {
	const int n = 1 << 10;
	const int blockSize = 1 << 10;
	float* x, * y, * dx, * dy;
	cudaMalloc(&dx, n * sizeof(float));
	cudaMalloc(&dy, n * sizeof(float));
	x = (float*)malloc(n * sizeof(float));
	y = (float*)malloc(n * sizeof(float));
	memset(y, 0, n * sizeof(float));
	for (int i = 0; i < n; ++i)x[i] = 1;
	cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);

	/*cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	const int num_trials = 1000;
	float total1 = 0, time = 0;
	for (int i = 0; i < num_trials; ++i) {
		cudaEventRecord(start, 0);*/
		scan1<< <1, BLOCK_SIZE/2 >> > (dx, dy);
	/*	cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		total1 += time;
	}
	cudaDeviceSynchronize();
	std::cout << "average duration =" << total1 << "\n";
	float total2 = 0;
	for (int i = 0; i < num_trials; ++i) {
		cudaEventRecord(start, 0);
		upsweep << <1, n / 2 >> > (dx, dy, n);
		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		total2 += time;
	}
	std::cout << "average duration =" << total2 << "\n";*/

	cudaMemcpy(y, dy, n * sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < n; ++i)
	//	if (y[i] != i * (i + 1) / 2)std::cout << "error";
	//for( int i=n-1;i>n-3;--i)
	std::cout << y[n - 1] + x[n - 1] << "\n";
	std::cout << std::endl;
}