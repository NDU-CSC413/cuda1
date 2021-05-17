#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>
#include <chrono>
/**
 * Matrix multiplication using shared memory.
 * The matrix is assumed to be square.
 */
using Duration = std::chrono::duration<double, std::milli>;

#define TIMEIT(dur,...)\
   {\
    auto start = std::chrono::high_resolution_clock::now();\
    __VA_ARGS__\
    auto end = std::chrono::high_resolution_clock::now();\
     dur = std::chrono::duration<double, std::milli>(end - start);\
}
#define BLOCK_SIZE 32
__global__ void mult(float* da, float* db, float* dc, int width) {

	int by= blockIdx.y;
	int bx = blockIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;
	__shared__ float sa[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float sb[BLOCK_SIZE][BLOCK_SIZE];
	float res = 0.0;
	int ntiles = width / BLOCK_SIZE;
	for (int b = 0; b < ntiles; ++b) {
		
		/* copy from memory to shared memory */
		sa[ty][tx] = da[row * width + b * BLOCK_SIZE + tx];
		sb[ty][tx] = db[(b * BLOCK_SIZE + ty) * width + col];
		
		__syncthreads();
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			res += sa[ty][k] * sb[k][tx];
		}
		__syncthreads();
	}
	dc[row* width + col] = res;
}


int main() {
	cudaEvent_t kernel_start,kernel_end;
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_end);


	float* a, * b, * c;
	float* da, * db, * dc;

	const int matrix_width = 1024;
	const int size = matrix_width * matrix_width;
	a = (float*)malloc(size * sizeof(float));
	b = (float*)malloc(size * sizeof(float));
	c = (float*)malloc(size * sizeof(float));
	for (int i = 0; i < size; ++i) {
		a[i] = 1;
		b[i] = 1;
	}
	cudaMalloc(&da, size * sizeof(float));
	cudaMalloc(&db, size * sizeof(float));
	cudaMalloc(&dc, size * sizeof(float));
	cudaMemcpy(da, a, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, size * sizeof(float), cudaMemcpyHostToDevice);
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize(matrix_width/ BLOCK_SIZE, matrix_width / BLOCK_SIZE);
	mult <<<gridSize, blockSize >> > (da, db, dc, matrix_width);
	float time = 0;
	float gpu_time = 0;
	const int num_trials = 500;
	for (int i = 0; i < num_trials; ++i) {
		cudaEventRecord(kernel_start,0);
		mult << <gridSize, blockSize >> > (da, db, dc, matrix_width);
		cudaEventRecord(kernel_end,0);
		cudaEventSynchronize(kernel_end);
		cudaEventElapsedTime(&time, kernel_start, kernel_end);
		gpu_time += time;
	}
	gpu_time /= num_trials;
	std::cout << "GPU  time " << gpu_time << '\n';
	cudaMemcpy(c, dc, size * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++) {
		if (c[i] != matrix_width) {
			std::cout << "error\n";
			break;
		}
		else c[i] = 0;
	}
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	Duration d;
	TIMEIT(d,
		for (int i = 0; i < matrix_width; ++i) {
			for (int j = 0; j < matrix_width; ++j)
				for (int k = 0; k < matrix_width; ++k)
					c[i * matrix_width + j] += a[i * matrix_width+ k] * b[matrix_width * k + j];
		}
	)
		
	std::cout << "CPU time " << d.count() << " milliseconds \n";
	std::cout << "gain = " << d.count() / gpu_time << "\n";
	free(a);
	free(b);
	free(c);

}