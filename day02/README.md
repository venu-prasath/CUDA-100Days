## Matrix Addition

This project demonstrates matrix addition using both CPU and GPU implementations in CUDA.

### Features
- **CPU Implementation**: A simple matrix addition function that runs on the CPU.
- **GPU Implementation**: A CUDA kernel that performs matrix addition in parallel using a 2D grid and block configuration.

### How It Works
1. Matrices `A` and `B` are initialized with random values.
2. The CPU implementation computes the result matrix `C` sequentially.
3. The GPU implementation computes the result matrix `C` in parallel using CUDA.
4. The results from the CPU and GPU are compared to ensure correctness.

### Key Concepts
- **CUDA Kernels**: The GPU kernel uses a 2D grid and block structure to parallelize the computation.
- **Memory Management**: Host and device memory are allocated and managed explicitly.
- **Boundary Checks**: The kernel ensures threads do not access out-of-bounds memory.

### How to Run
1. Compile the CUDA program:
   ```bash
   nvcc matrix_add.cu -o matrix_add
   ```

   or 

   ```
   cd build && cmake .. && make
   ```

2. Run the executable:
```bash
./matrix_add
```

### Benchmarking
The program benchmarks the performance of both CPU and GPU implementations by measuring their execution times:

- **CPU Benchmarking**: Uses the `<chrono>` library to measure the time taken for matrix addition on the CPU.
- **GPU Benchmarking**: Uses CUDA events to measure the time taken for matrix addition on the GPU.

The results are displayed in milliseconds, allowing for a direct comparison of the performance between the two implementations.

### Output
The program will output the time taken for both CPU and GPU implementations and verify that the results are correct.
