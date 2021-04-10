
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>



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
	mult << <gridSize, blockSize >> > (da, db, dc, matrix_width);
	float time = 0;
	float total = 0;

	for (int i = 0; i < 500; ++i) {
		cudaEventRecord(kernel_start,0);
		mult << <gridSize, blockSize >> > (da, db, dc, matrix_width);
		cudaEventRecord(kernel_end,0);
		cudaEventSynchronize(kernel_end);
		cudaEventElapsedTime(&time, kernel_start, kernel_end);
		total += time;
	}
	std::cout << "average time " << total/500 << '\n';
	cudaMemcpy(c, dc, size * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 64; i++)
		std::cout << c[i] << ' ';
	std::cout << std::endl;
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	free(a);
	free(b);
	free(c);

}