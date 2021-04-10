#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void mmult(float* a, float* b, float* ab, size_t width)
{
    // calculate the row & column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float result = 0;

    // do dot product between row of a and column of b
    for (int k = 0; k < width; ++k)
    {
        result += a[row * width + k] * b[k * width + col];
    }

    // write out this thread's result
    ab[row * width + col] = result;
}
__global__ void kernel2(float* da, float* db, float* dc, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float result = 0;
    for (int k = 0; k < width; ++k) 
    {
        result += da[row * width + k] * db[k * width + col];
    }
    dc[row * width + col] = result;
}

void callKernel(float* da, float* db, float* dc, int width, dim3 block_size) {
    dim3 blocksPerGrid(width / block_size.x, width / block_size.y);
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);
    kernel2 << <blocksPerGrid, block_size >> > (da, db, dc, width);
    float time = 0;
    float total = 0;
    for (int i = 0; i < 100; ++i) {
        cudaEventRecord(kernel_start);
        mmult<< <blocksPerGrid, block_size >> > (da, db, dc, width);
        cudaEventRecord(kernel_end);
        cudaEventSynchronize(kernel_end);
        cudaEventElapsedTime(&time, kernel_start, kernel_end);
        total += time;
    }
    std::cout << "time " << total/100 << '\n';
    total = 0.;
    for (int i = 0; i < 100; ++i) {
        cudaEventRecord(kernel_start);
        kernel2 << <blocksPerGrid, block_size >> > (da, db, dc, width);
        cudaEventRecord(kernel_end);
        cudaEventSynchronize(kernel_end);
        cudaEventElapsedTime(&time, kernel_start, kernel_end);
        total += time;
    }
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

    /*dim3 threadsPerBlock (matrix_w,matrix_w);
    kernel1<<<1,threadsPerBlock>>>(da,db,dc,matrix_w);*/
    /* total number of threads per block is 1024 which is the maximum */
    dim3 threadsPerBlock(16, 16);
    callKernel(da, db, dc, matrix_w, threadsPerBlock);
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