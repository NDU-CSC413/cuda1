#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
/**
 *  mat_mult()->__global__ void
 * Matrix multiplication without using shared memory
 * @param da
 * @param db
 * @param dc
 * @param width
 * @return 
 */
__global__ void mat_mult(float* da, float* db, float* dc, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float result = 0;
    for (int k = 0; k < width; ++k) 
    {
        result += da[row * width + k] * db[k * width + col];
    }
    dc[row * width + col] = result;
}

void time_kernel(float* da, float* db, float* dc, int width,
                         dim3 blocks_per_grid,dim3 threads_per_block) {
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);
    /* warmup call*/
    mat_mult <<<blocks_per_grid, threads_per_block >> > (da, db, dc, width);
    float time = 0;
    float total = 0;
   
    for (int i = 0; i < 100; ++i) {
        cudaEventRecord(kernel_start);
        mat_mult << <blocks_per_grid, threads_per_block>> > (da, db, dc, width);
        cudaEventRecord(kernel_end);
        cudaEventSynchronize(kernel_end);
        cudaEventElapsedTime(&time, kernel_start, kernel_end);
        total += time;
    }
    /* average time in milliseconds */
    std::cout << "time " << total / 100 << '\n';

}
int main() {
    const int matrix_w = 1024;
    const int msize = matrix_w * matrix_w;
    float* a, * b, * c;

    float* da, * db, * dc;
    a = (float*)malloc(msize * sizeof(float));
    b = (float*)malloc(msize * sizeof(float));
    c = (float*)malloc(msize * sizeof(float));
    for (int i = 0; i < msize; ++i) {
        a[i] = 1;
        b[i] = 1;
    }

    cudaMalloc(&da, msize * sizeof(float));
    cudaMalloc(&db, msize * sizeof(float));
    cudaMalloc(&dc, msize * sizeof(float));
    cudaMemcpy(da, a, msize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, msize * sizeof(float), cudaMemcpyHostToDevice);

    
    /* total number of threads per block is 1024 which is the maximum */
    dim3 threads_per_block(32, 32);
    dim3 blocks_per_grid(matrix_w / threads_per_block.x, matrix_w/ threads_per_block.y);
    time_kernel(da, db, dc, matrix_w, blocks_per_grid,threads_per_block);
    cudaMemcpy(c, dc, msize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < msize; ++i)
        if (c[i] != 1024)std::cout << "ERROR\n";
    //std::cout << c[i] << ' ';
    std::cout << std::endl;
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);


}