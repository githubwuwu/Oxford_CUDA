//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <cuda_runtime.h>
//
// kernel routine
// 

__global__ void my_first_kernel(float *x)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;

	x[tid] = (float)threadIdx.x;
}


//
// main code
//

int main(int argc, char **argv)
{
	float *h_x, *d_x;
	int   nblocks, nthreads, nsize, n;

	// 初始化
	int  devID= findCudaDevice(argc, (const char**)argv);
	cudaDeviceProp deviceProps;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
	printf("CUDA device [%s]\n", deviceProps.name);

	// set number of blocks, and threads per block

	nblocks = 0;
	nthreads = 8;
	nsize = nblocks*nthreads;

	// allocate memory for array

	h_x = (float *)malloc(nsize*sizeof(float));
	checkCudaErrors(cudaMalloc(&d_x, nsize*sizeof(float)));

	// execute kernel

	my_first_kernel << <nblocks, nthreads >> >(d_x);
	getLastCudaError("my_first_kernel execution failed\n");

	// copy back results and print them out

	checkCudaErrors(cudaMemcpy(h_x, d_x, nsize*sizeof(float), cudaMemcpyDeviceToHost));

	for (n = 0; n<nsize; n++) printf(" n,  x  =  %d  %f \n", n, h_x[n]);

	// free memory 

	checkCudaErrors(cudaFree(d_x));
	free(h_x);

	// CUDA exit -- needed to flush printf write buffer
	//cudaDeviceReset()是为了刷新printf??清空之前所关联过得所有资源。
	//或者cudaDeviceSynchronize()
	cudaDeviceReset();

	return 0;

	//如果使用统一内存的话 只需要申请一次内存
	//checkCudaErrors(cudaMallocManaged(&x, nsize*sizeof(float)));
	// cudaDeviceSynchronize();
}
