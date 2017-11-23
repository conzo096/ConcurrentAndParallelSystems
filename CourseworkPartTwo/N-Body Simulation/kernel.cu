
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#define ELEMENTS 20
#define PARTICLES 2000

void cuda_info()
{
	int device;
	cudaGetDevice(&device);

	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);

	std::cout << "Name: " << properties.name << std::endl;
	std::cout << "CUDA Capability: " << properties.major <<  "." << properties.minor << std::endl;
	std::cout << "Cores: " << properties.multiProcessorCount << std::endl;
	std::cout << "Memory: : " << properties.totalGlobalMem / (1024*1024) << "MB" << std::endl;
	std::cout << "Clock freq: " << properties.clockRate/1000 << "MHz" << std::endl;
}

__global__ void vecAdd(const int *A, const int *B, int *C)
{
	unsigned int block_idx = blockIdx.x;
	unsigned int thread_idx = threadIdx.x;
	unsigned int block_dim = blockDim.x;

	unsigned int idx = (block_idx * block_dim) + thread_idx;

	C[idx] = A[idx] + B[idx];
}



// Updates particles based on their 
__global__ void Update(const float *mass, float *px, float *py, float *pz,
						float *vx, float *vy, float *vz,	
						const float *fx, const float *fy, const float *fz)
{
	unsigned int block_idx = blockIdx.x;
	unsigned int thread_idx = threadIdx.x;
	unsigned int block_dim = blockDim.x;
	unsigned int idx = (block_idx * block_dim) + thread_idx;

	vx[idx] += fx[idx] / mass[idx];
	vy[idx] += fy[idx] / mass[idx];
	vz[idx] += fz[idx] / mass[idx];

	px[idx] = 0.16 * vx[idx];
	py[idx] = 0.16 * vy[idx];
	pz[idx] = 0.16 * vz[idx];

}


int main()
{
	// Initialise CUDA
	cudaSetDevice(0);
	cuda_info();

	// Create host memory.
	auto data_size = sizeof(float) * PARTICLES;

	float px[PARTICLES];
	float py[PARTICLES];
	float pz[PARTICLES];

	float vx[PARTICLES];
	float vy[PARTICLES];
	float vz[PARTICLES];

	float fx[PARTICLES];
	float fy[PARTICLES];
	float fz[PARTICLES];

	float mass[PARTICLES];



	// Populate input data.
	for (unsigned int i = 0; i < ELEMENTS; i++)
	{
		px[i] = i;
		py[i] = i;
		pz[i] = i;
		mass[i] = rand() % 255 + 10;

	}
	// Declare buffers.
	float *buffer_px, *buffer_py, *buffer_pz;
	float *buffer_vx, *buffer_vy, *buffer_vz;
	float *buffer_fx, *buffer_fy, *buffer_fz;
	float *buffer_mass;



	// init buffers.
	cudaMalloc((void**)&buffer_px, data_size);
	cudaMalloc((void**)&buffer_py, data_size);
	cudaMalloc((void**)&buffer_pz, data_size);

	cudaMalloc((void**)&buffer_vx, data_size);
	cudaMalloc((void**)&buffer_vy, data_size);
	cudaMalloc((void**)&buffer_vz, data_size);

	cudaMalloc((void**)&buffer_fx, data_size);
	cudaMalloc((void**)&buffer_fy, data_size);
	cudaMalloc((void**)&buffer_fz, data_size);

	cudaMalloc((void**)&buffer_mass, data_size);

	cudaMemcpy(buffer_px, &px[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_py, &py[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_pz, &pz[0], data_size, cudaMemcpyHostToDevice);

	cudaMemcpy(buffer_vx, &vx[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_vy, &vy[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_vz, &vz[0], data_size, cudaMemcpyHostToDevice);

	cudaMemcpy(buffer_fx, &fx[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_fy, &fy[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_fz, &fz[0], data_size, cudaMemcpyHostToDevice);

	cudaMemcpy(buffer_mass, &mass[0], data_size, cudaMemcpyHostToDevice);


	// Now loop.
	for (int i = 0; i < 10; i++)
	{
		// Execute kernel.
		Update <<<PARTICLES / 1024, 1024 >> >(buffer_mass,buffer_px,buffer_py,buffer_pz,
											buffer_vx, buffer_vy, buffer_vz,
											buffer_fx, buffer_fy, buffer_fz);

		// Wait for kernel to complete.
		cudaDeviceSynchronize();
		cudaMemcpy(&px[0], buffer_px, data_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(&py[0], buffer_py, data_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(&pz[0], buffer_pz, data_size, cudaMemcpyDeviceToHost);

		cudaMemcpy(&vx[0], buffer_vx, data_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(&vy[0], buffer_vy, data_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(&vz[0], buffer_vz, data_size, cudaMemcpyDeviceToHost);
	}

	// Read output.


	// Clean up resources.
	cudaFree(buffer_mass);
	cudaFree(buffer_px);
	cudaFree(buffer_py);
	cudaFree(buffer_pz);

	cudaFree(buffer_fx);
	cudaFree(buffer_fy);
	cudaFree(buffer_fz);

	cudaFree(buffer_vx);
	cudaFree(buffer_vy);
	cudaFree(buffer_vz);
	return 0;

}



















//int main()
//{
//	// Initialise CUDA
//	cudaSetDevice(0);
//	cuda_info();
//
//	// Create host memory.
//	auto data_size = sizeof(int) * ELEMENTS;
//	// input vectors.
//	std::vector<int> A(ELEMENTS);
//	std::vector<int> B(ELEMENTS);
//	// output vector.
//	std::vector<int> C(ELEMENTS);
//
//	// Populate input data.
//	for (unsigned int i = 0; i < ELEMENTS; i++)
//		A[i] = B[i] = i;
//
//	// Declare buffers.
//	int *buffer_A, *buffer_B, *buffer_C;
//
//	// init buffers.
//	cudaMalloc((void**)&buffer_A, data_size);
//	cudaMalloc((void**)&buffer_B, data_size);
//	cudaMalloc((void**)&buffer_C, data_size);
//
//	cudaMemcpy(buffer_A, &A[0], data_size, cudaMemcpyHostToDevice);
//	cudaMemcpy(buffer_B, &B[0], data_size, cudaMemcpyHostToDevice);
//
//	vecAdd<<<ELEMENTS/1024, 1024>>>(buffer_A, buffer_B, buffer_C);
//	// Wait for kerner to complete.
//	cudaDeviceSynchronize();
//
//	// Read output.
//	cudaMemcpy(&C[0], buffer_C, data_size, cudaMemcpyDeviceToHost);
//
//
//	// Clean up resources.
//	cudaFree(buffer_A);
//	cudaFree(buffer_B);
//	cudaFree(buffer_C);
//	return 0;
//
//}