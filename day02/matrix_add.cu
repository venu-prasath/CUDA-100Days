#include <iostream>
#include <chrono> 

// Matrix addition on GPU kernel
__global__
void matrixAddGPU(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}

// Matrix addition in CPU
void matrixAddCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N * N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    std::cout << "Matrix Addition using CUDA" << std::endl;
    int N = 1024;
    int size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
    
    // Initialize matrices A and B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Function to print a matrix
    auto printMatrix = [](const float *matrix, int N, const std::string &name, int limit) {
        std::cout << "First " << limit << " elements of Matrix " << name << ":" << std::endl;
        for (int i = 0; i < limit; ++i) {
            std::cout << matrix[i] << " ";
        }
        std::cout << std::endl;
    };

    // Print the first 10 elements of the matrices A and B
    printMatrix(h_A, N, "A", 10);
    printMatrix(h_B, N, "B", 10);

    // Benchmark CPU implementation
    auto startCPU = std::chrono::high_resolution_clock::now();
    matrixAddCPU(h_A, h_B, h_C, N);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endCPU - startCPU;
    std::cout << "CPU Time: " << cpuDuration.count() << " ms" << std::endl;

    // Print the first 10 elements of the result matrix C
    printMatrix(h_C, N, "C", 10);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Benchmark GPU implementation
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    cudaEventRecord(startGPU);
    matrixAddGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopGPU);

    cudaEventSynchronize(stopGPU);
    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startGPU, stopGPU);
    std::cout << "GPU Time: " << gpuDuration << " ms" << std::endl;

    // Print the first 10 elements of the result matrix C from GPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    printMatrix(h_C, N, "GPU", 10);

    // Verify the result
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": CPU " << h_C[i] << ", GPU " << (h_A[i] + h_B[i]) << std::endl;
            return -1;
        }
    }
    std::cout << "Verification passed!" << std::endl;
    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);
    
    std::cout << "Matrix addition completed successfully." << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    std::cout << "Memory allocated and freed successfully." << std::endl;
    return 0;
}