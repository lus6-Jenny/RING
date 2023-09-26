#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <algorithm>
#include <iomanip>
#include <time.h>

#define USE_SCALAR_IMPLEMENTATION
#define COMPUTE_V_AS_MATRIX
#define COMPUTE_U_AS_MATRIX

#include "Singular_Value_Decomposition_Preamble.hpp"
#include "svd3_cuda.h"

#define VERIFY_RESULTS

#define USE_SOA	// use Structure of Arrays for matrix attributes or Array of structures

__global__ void svd3_SOA(float* input, float* ouputdata, int testsize)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= testsize) return;

	svd(
#ifdef USE_SOA	
		input[tid + 0 * testsize], input[tid + 1 * testsize], input[tid + 2 * testsize],
		input[tid + 3 * testsize], input[tid + 4 * testsize], input[tid + 5 * testsize],
		input[tid + 6 * testsize], input[tid + 7 * testsize], input[tid + 8 * testsize],

		ouputdata[tid + 0 * testsize], ouputdata[tid + 1 * testsize], ouputdata[tid + 2 * testsize],
		ouputdata[tid + 3 * testsize], ouputdata[tid + 4 * testsize], ouputdata[tid + 5 * testsize],
		ouputdata[tid + 6 * testsize], ouputdata[tid + 7 * testsize], ouputdata[tid + 8 * testsize],

		ouputdata[tid + 9 * testsize], ouputdata[tid + 10 * testsize], ouputdata[tid + 11 * testsize],

		ouputdata[tid + 12 * testsize], ouputdata[tid + 13 * testsize], ouputdata[tid + 14 * testsize],
		ouputdata[tid + 15 * testsize], ouputdata[tid + 16 * testsize], ouputdata[tid + 17 * testsize],
		ouputdata[tid + 18 * testsize], ouputdata[tid + 19 * testsize], ouputdata[tid + 20 * testsize]
#else 
		input[tid * 9 + 0], input[tid * 9 + 1], input[tid * 9 + 2], 
		input[tid * 9 + 3], input[tid * 9 + 4], input[tid * 9 + 5], 
		input[tid * 9 + 6], input[tid * 9 + 7], input[tid * 9 + 8],

		ouputdata[tid * 21 + 0], ouputdata[tid * 21 + 1], ouputdata[tid * 21 + 2],
		ouputdata[tid * 21 + 3], ouputdata[tid * 21 + 4], ouputdata[tid * 21 + 5],
		ouputdata[tid * 21 + 6], ouputdata[tid * 21 + 7], ouputdata[tid * 21 + 8],

		ouputdata[tid * 21 + 9], ouputdata[tid * 21 + 10], ouputdata[tid * 21 + 11],

		ouputdata[tid * 21 + 12], ouputdata[tid * 21 + 13], ouputdata[tid * 21 + 14],
		ouputdata[tid * 21 + 15], ouputdata[tid * 21 + 16], ouputdata[tid * 21 + 17],
		ouputdata[tid * 21 + 18], ouputdata[tid * 21 + 19], ouputdata[tid * 21 + 20]
#endif // USE_SOA	
	);
}

__global__ void svd3_AOS_shared(float* input, float* ouputdata, int testsize)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= testsize) return;

	int threadPerBlock = min(blockDim.x, testsize);

	if (tid >= testsize / blockDim.x * blockDim.x) threadPerBlock = testsize % blockDim.x;
	
	__shared__ un sArray[504 * 21];

	// load to shared memory
	for (int i = 0; i < 9; i++)
	{
		int pos = i * threadPerBlock + threadIdx.x;
		sArray[pos / 9 * 21 + pos % 9].f = input[blockDim.x * 9 * blockIdx.x + pos];
	}

	__syncthreads();	// sync after load 

	svd(sArray[threadIdx.x * 21 + 0].f,  sArray[threadIdx.x * 21 + 1].f,  sArray[threadIdx.x * 21 + 2].f,
		sArray[threadIdx.x * 21 + 3].f,  sArray[threadIdx.x * 21 + 4].f,  sArray[threadIdx.x * 21 + 5].f,
		sArray[threadIdx.x * 21 + 6].f,  sArray[threadIdx.x * 21 + 7].f,  sArray[threadIdx.x * 21 + 8].f,
		sArray[threadIdx.x * 21 + 0].f,  sArray[threadIdx.x * 21 + 1].f,  sArray[threadIdx.x * 21 + 2].f,
		sArray[threadIdx.x * 21 + 3].f,  sArray[threadIdx.x * 21 + 4].f,  sArray[threadIdx.x * 21 + 5].f,
		sArray[threadIdx.x * 21 + 6].f,  sArray[threadIdx.x * 21 + 7].f,  sArray[threadIdx.x * 21 + 8].f,
		sArray[threadIdx.x * 21 + 9].f,  sArray[threadIdx.x * 21 + 10].f, sArray[threadIdx.x * 21 + 11].f,
		sArray[threadIdx.x * 21 + 12].f, sArray[threadIdx.x * 21 + 13].f, sArray[threadIdx.x * 21 + 14].f,
		sArray[threadIdx.x * 21 + 15].f, sArray[threadIdx.x * 21 + 16].f, sArray[threadIdx.x * 21 + 17].f,
		sArray[threadIdx.x * 21 + 18].f, sArray[threadIdx.x * 21 + 19].f, sArray[threadIdx.x * 21 + 20].f
	);
	
	__syncthreads();	// sync before store 

	for (int i = 0; i < 21; i++)
		ouputdata[blockDim.x * 21 * blockIdx.x + i * threadPerBlock + threadIdx.x] = sArray[i * threadPerBlock + threadIdx.x].f;
}


void runCudaPart(float* input, float& output, int n)
{
	float* d_answer;
	cudaMalloc(&d_answer, 21 * sizeof(float) * n);

	float* d_input;
	cudaMalloc(&d_input, 9 * sizeof(float) * n);

	cudaMemcpy(d_input, input, 9 * sizeof(float) * n, cudaMemcpyHostToDevice);

	int threads = 504;
	int pblks = int(n / threads) + 1;

	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

#ifdef USE_SOA
	svd3_SOA << <pblks, threads >> >(d_input, d_answer, n);
#else 
	svd3_AOS_shared << <pblks, threads >> >(d_input, d_answer, n);
#endif // USE_SOA

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms\n", elapsedTime);

	cudaMemcpy(&output, d_answer, 21 * sizeof(float) * n, cudaMemcpyDeviceToHost);
	cudaFree(d_answer);
	cudaDeviceSynchronize();
}

int main()
{
	// Load data
	std::ifstream myfile;
	myfile.open("Dataset_1M.txt");
	int testsSize;
	myfile >> testsSize;
	testsSize = testsSize;
	std::cout << "dataset size: " << testsSize << std::endl;

	float* input = (float*)malloc(sizeof(float) * 9 * testsSize);
	int count = 0;
	for (int i = 0; i < testsSize; i++)
		for (int j = 0; j < 9; j++) myfile >> input[count++];
	myfile.close();

	float* result = (float*)malloc(sizeof(float) * 21 * testsSize);

	// CUDA SVD 3x3
	runCudaPart(input, *result, testsSize);

#ifdef VERIFY_RESULTS
	// run CPU 3x3 to verify results
	float a11, a12, a13, a21, a22, a23, a31, a32, a33;
	for (int i = 0; i < testsSize; i++)
	{
#ifdef USE_SOA
		a11 = input[i + 0 * testsSize]; a12 = input[i + 1 * testsSize]; a13 = input[i + 2 * testsSize];
		a21 = input[i + 3 * testsSize]; a22 = input[i + 4 * testsSize]; a23 = input[i + 5 * testsSize];
		a31 = input[i + 6 * testsSize]; a32 = input[i + 7 * testsSize]; a33 = input[i + 8 * testsSize];
#else
		a11 = input[i * 9 + 0]; a12 = input[i * 9 + 1]; a13 = input[i * 9 + 2];
		a21 = input[i * 9 + 3]; a22 = input[i * 9 + 4]; a23 = input[i * 9 + 5];
		a31 = input[i * 9 + 6]; a32 = input[i * 9 + 7]; a33 = input[i * 9 + 8];
#endif // USE_SOA

#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"

		ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = a11;) ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = a21;) ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = a31;)
		ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = a12;) ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = a22;) ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = a32;)
		ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = a13;) ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = a23;) ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = a33;)

#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp" 

		bool isWrong = false;
#ifdef USE_SOA
		if (fabsf(result[i +  0 * testsSize] - Su11.f) > 1e-6) isWrong = true;
		if (fabsf(result[i +  1 * testsSize] - Su12.f) > 1e-6) isWrong = true;
		if (fabsf(result[i +  2 * testsSize] - Su13.f) > 1e-6) isWrong = true;
		if (fabsf(result[i +  3 * testsSize] - Su21.f) > 1e-6) isWrong = true;
		if (fabsf(result[i +  4 * testsSize] - Su22.f) > 1e-6) isWrong = true;
		if (fabsf(result[i +  5 * testsSize] - Su23.f) > 1e-6) isWrong = true;
		if (fabsf(result[i +  6 * testsSize] - Su31.f) > 1e-6) isWrong = true;
		if (fabsf(result[i +  7 * testsSize] - Su32.f) > 1e-6) isWrong = true;
		if (fabsf(result[i +  8 * testsSize] - Su33.f) > 1e-6) isWrong = true;
		if (fabsf(result[i +  9 * testsSize] - Sa11.f) > 1e-6) isWrong = true;
		if (fabsf(result[i + 10 * testsSize] - Sa22.f) > 1e-6) isWrong = true;
		if (fabsf(result[i + 11 * testsSize] - Sa33.f) > 1e-6) isWrong = true;
		if (fabsf(result[i + 12 * testsSize] - Sv11.f) > 1e-6) isWrong = true;
		if (fabsf(result[i + 13 * testsSize] - Sv12.f) > 1e-6) isWrong = true;
		if (fabsf(result[i + 14 * testsSize] - Sv13.f) > 1e-6) isWrong = true;
		if (fabsf(result[i + 15 * testsSize] - Sv21.f) > 1e-6) isWrong = true;
		if (fabsf(result[i + 16 * testsSize] - Sv22.f) > 1e-6) isWrong = true;
		if (fabsf(result[i + 17 * testsSize] - Sv23.f) > 1e-6) isWrong = true;
		if (fabsf(result[i + 18 * testsSize] - Sv31.f) > 1e-6) isWrong = true;
		if (fabsf(result[i + 19 * testsSize] - Sv32.f) > 1e-6) isWrong = true;
		if (fabsf(result[i + 20 * testsSize] - Sv33.f) > 1e-6) isWrong = true;
#else 
		if (fabsf(result[i * 21 +  0] - Su11.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 +  1] - Su12.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 +  2] - Su13.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 +  3] - Su21.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 +  4] - Su22.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 +  5] - Su23.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 +  6] - Su31.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 +  7] - Su32.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 +  8] - Su33.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 +  9] - Sa11.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 + 10] - Sa22.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 + 11] - Sa33.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 + 12] - Sv11.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 + 13] - Sv12.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 + 14] - Sv13.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 + 15] - Sv21.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 + 16] - Sv22.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 + 17] - Sv23.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 + 18] - Sv31.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 + 19] - Sv32.f) > 1e-6) isWrong = true;
		if (fabsf(result[i * 21 + 20] - Sv33.f) > 1e-6) isWrong = true;
#endif // USE_SOA

		if (isWrong)
		{
			std::cout << "------------------------------- " << i << std::endl;
			std::cout << "input\n";
			std::cout << a11 << " " << a12 << " " << a13 << "\n";
			std::cout << a21 << " " << a22 << " " << a23 << "\n";
			std::cout << a31 << " " << a32 << " " << a33 << "\n";
			std::cout << "cuda result\n";								   
			std::cout << result[i * 21 + 0] << " " << result[i * 21 + 1] << " " << result[i * 21 + 2] << "\n";
			std::cout << result[i * 21 + 3] << " " << result[i * 21 + 4] << " " << result[i * 21 + 5] << "\n";
			std::cout << result[i * 21 + 6] << " " << result[i * 21 + 7] << " " << result[i * 21 + 8] << "\n";
			std::cout << result[i * 21 + 9] << " " << result[i * 21 + 10] << " " << result[i * 21 + 11] << "\n";
			std::cout << result[i * 21 + 12] << " " << result[i * 21 + 13] << " " << result[i * 21 + 14] << "\n";
			std::cout << result[i * 21 + 15] << " " << result[i * 21 + 16] << " " << result[i * 21 + 17] << "\n";
			std::cout << result[i * 21 + 18] << " " << result[i * 21 + 19] << " " << result[i * 21 + 20] << "\n";
			std::cout << "CPU result\n";
			std::cout << std::setw(12) << Su11.f << ",  " << std::setw(12) << Su12.f << ",  " << std::setw(12) << Su13.f << std::endl;
			std::cout << std::setw(12) << Su21.f << ",  " << std::setw(12) << Su22.f << ",  " << std::setw(12) << Su23.f << std::endl;
			std::cout << std::setw(12) << Su31.f << ",  " << std::setw(12) << Su32.f << ",  " << std::setw(12) << Su33.f << std::endl;
			std::cout << std::setw(12) << Sa11.f << ",  " << std::setw(12) << Sa22.f << ",  " << std::setw(12) << Sa33.f << std::endl;
			std::cout << std::setw(12) << Sv11.f << ",  " << std::setw(12) << Sv12.f << ",  " << std::setw(12) << Sv13.f << std::endl;
			std::cout << std::setw(12) << Sv21.f << ",  " << std::setw(12) << Sv22.f << ",  " << std::setw(12) << Sv23.f << std::endl;
			std::cout << std::setw(12) << Sv31.f << ",  " << std::setw(12) << Sv32.f << ",  " << std::setw(12) << Sv33.f << std::endl;
		}
	}
#endif

	std::cout << "Test is finished\n";

	free(result);

	return 0;
}
