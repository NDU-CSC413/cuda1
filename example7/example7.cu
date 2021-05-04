#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using Duration = std::chrono::duration<double,std::milli>;

#define TIMEIT(dur,...)\
   {\
    auto start = std::chrono::high_resolution_clock::now();\
    __VA_ARGS__\
    auto end = std::chrono::high_resolution_clock::now();\
     dur = std::chrono::duration<double, std::milli>(end - start);\
}
__host__ void add(float *a,float *b,float *c,int n){
    for(int i=0;i<n;++i)
        c[i]=a[i]+b[i];
}
__global__ void kernel(float* a, float* b, float* c,int n) {
	int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id<n)
	  c[id] = a[id] + b[id];
}
__managed__ int gd=23;
__global__ void print(int *gd){
     *gd=gridDim.x;
}

int main() {
	int N = 1<<25;
	float* a, * b, * c;
	float* da, * db, * dc;
  /* allocate memory on host */
	a = (float*)malloc(N * sizeof(float));
	b = (float*)malloc(N * sizeof(float));
	c = (float*)malloc(N * sizeof(float));
  /* allocate memory on device */
	cudaMalloc(&da, N * sizeof(float));
	cudaMalloc(&db, N * sizeof(float));
	cudaMalloc(&dc, N * sizeof(float));
  /* initialize the arrays a and b */
	for (int i = 0; i < N; ++i) {
		a[i] = i;
		b[i] = 2 * i;
	}
  Duration d;
  TIMEIT(d,
   add(a,b,c,N);
  )
  std::cout<<"CPU duration ="<<d.count()<<"\n";
  /* copy arrays a and b to device */
	cudaMemcpy(da, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, N * sizeof(float), cudaMemcpyHostToDevice);
/* divide the load into num_blocks */
  int num_blocks=N/1024;
  /* declare CUDA events */
  cudaEvent_t launch_begin,launch_end;
  /* create event objects */
  cudaEventCreate(&launch_begin);
  cudaEventCreate(&launch_end);
  float average_time=0;
  /* launch the kernel num_launches times to
   * get a more accurate estimate */
   const int num_launches=100;
  for(int i=0;i<num_launches;++i){
      cudaEventRecord(launch_begin,0);
	    kernel << <num_blocks, 1024>> > (da, db, dc,N);
      cudaEventRecord(launch_end,0);
      cudaEventSynchronize(launch_end);
      float time=0;
      cudaEventElapsedTime(&time,launch_begin,launch_end);
      average_time+=time;
  }

  /* copy result to host */
	cudaMemcpy(c, dc, N * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout<<"average time= "<<average_time/num_launches<<"\n";


  /* print the last 10 elements */
	for (int i = 0; i < 10; ++i)
		std::cout << c[2047-i] << ' ';
	std::cout << std::endl;
	//dim3 ui3(1,2,3);
	
	uint3 ui3=make_uint3(1,2,3);

	std::cout<<ui3.x<<"\n";
	std::cout<<ui3.y<<"\n";
	std::cout<<ui3.z<<"\n";
	print<<<6,1>>>(&gd);
	cudaDeviceSynchronize();
	std::cout<<"griddim ="<<gd<<"\n";
	/* free memory on host and device */
	free(a);
	free(b);
	free(c);
	cudaFree(db);
	cudaFree(dc);
	cudaFree(da);

}