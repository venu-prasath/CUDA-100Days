#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
	std::cout << "CUDA Vector Addition" << std::endl;
    int N = 10000;
    size_t size = N * sizeof(float);

	// Allocate host memory
	float* h_A = (float*)malloc(size);
	float* h_B = (float*)malloc(size);
	float* h_C = (float*)malloc(size);

	// Initialize input vectors
	for (int i = 0; i < N; i++) {
		h_A[i] = static_cast<float>(i);
		h_B[i] = static_cast<float>(i * 2);
	}

	// Allocate device memory
	float* d_A, * d_B, * d_C;
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	// Copy data from host to device
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// Launch kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);

	// Check for errors in kernel launch
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Synchronize device
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to synchronize after kernel launch (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy result from device to host
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	// Verify result
	for (int i = 0; i < N; i++) {
		if (h_C[i] != h_A[i] + h_B[i]) {
			fprintf(stderr, "Error: h_C[%d] = %f, expected %f\n", i, h_C[i], h_A[i] + h_B[i]);
			exit(EXIT_FAILURE);
		}
	}

	std::cout << "Vector addition completed successfully!" << std::endl;


	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	return 0;
}