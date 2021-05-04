#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define THREADS_PER_BLOCK 1024

__global__ void reduction1(int* da, int* db) {
	__shared__ int sdata[THREADS_PER_BLOCK];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = da[i];
	__syncthreads();
	// do reduction in shared mem
	
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		if (tid % (2 * stride) == 0) {
			sdata[tid] += sdata[tid + stride];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) db[blockIdx.x] = sdata[0];
	
}
__global__ void reduction2(int* da, int* db) {
	__shared__ int sdata[THREADS_PER_BLOCK];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = da[i];
	__syncthreads();
	// do reduction in shared mem

	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		int index = 2 * stride * tid;
		if (index<blockDim.x) {
			sdata[index] += sdata[index + stride];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) db[blockIdx.x] = sdata[0];

}
__global__ void dot_product(int* da, int* dc,int *res) {
	//__shared__ int s_adata[THREADS_PER_BLOCK];
	//__shared__ int s_cdata[THREADS_PER_BLOCK];
	__shared__ int tmp[THREADS_PER_BLOCK];


	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	//s_adata[tid] = da[i];
	//s_cdata[tid] = dc[i];
	tmp[tid] += da[i]*dc[i];
	__syncthreads();
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		if (tid % (2 * stride) == 0) {
			tmp[tid] += tmp[tid + stride];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) res[blockIdx.x] = tmp[0];
}

int main() {
	/* we are using a 1-d grid. Max grid size is 65536 x 65536 x 65536*/
	const int n = 1 << 26;
	int *a;
	int *c;
	const int blocks_per_grid = n/THREADS_PER_BLOCK;
	
	int b[blocks_per_grid];
	a = (int*)malloc(n * sizeof(int));
	c = (int*)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) {
		a[i] = 2; c[i] = 3;
	}
	int* da,*dc;
	int* db;
	
	std::cout << "blocks per grid "<<blocks_per_grid << "\n";
	cudaMalloc(&da, n * sizeof(int));
	cudaMalloc(&dc, n * sizeof(int));
	cudaMalloc(&db, blocks_per_grid * sizeof(int));
	cudaMemcpy(da, a, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dc, c, n * sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	const int num_trials = 500;
	float total1 = 0, time = 0;
	for (int i = 0; i < num_trials; ++i) {
		cudaEventRecord(start, 0);
		reduction1<<<blocks_per_grid, THREADS_PER_BLOCK >> > (da, db);
		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time,start,end);
		total1 += time;
	}
	std::cout << "average duration =" << total1 / num_trials << "\n";
	float total2 = 0.;
	for (int i = 0; i < num_trials; ++i) {
		cudaEventRecord(start, 0);
		reduction2 << <blocks_per_grid, THREADS_PER_BLOCK >> > (da, db);
		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		total2 += time;
	}
	std::cout << "average duration(complete warps) =" << total2 / num_trials << "\n";
	std::cout << "gain " << total1 / total2 << "\n";

	cudaMemcpy(b, db, blocks_per_grid * sizeof(int), cudaMemcpyDeviceToHost);
	int sum = 0;
	for (int i = 0; i <blocks_per_grid; ++i)
		sum += b[i];
	std::cout <<"total= "<<sum<<"\n";
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);



}
