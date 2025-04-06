# CUDA Vector Addition

This folder contains the `vector_add_.cu` program, which demonstrates vector addition using CUDA.

## Program Overview

The program performs the following steps:
1. Allocates memory for two input vectors (`A` and `B`) and one output vector (`C`) on both the host (CPU) and the device (GPU).
2. Initializes the input vectors on the host.
3. Copies the input vectors from the host to the device.
4. Launches a CUDA kernel (`vectorAdd`) to compute the element-wise sum of the input vectors.
5. Copies the result vector from the device back to the host.
6. Verifies the correctness of the result on the host.
7. Frees the allocated memory on both the host and the device.

## Key CUDA Concepts

- **Memory Management**: The program uses `cudaMalloc` and `cudaMemcpy` for device memory allocation and data transfer between host and device.
- **Kernel Launch**: The `vectorAdd` kernel is launched with a grid and block configuration to parallelize the computation.
- **Error Handling**: The program checks for errors after kernel launches and device synchronization using `cudaGetLastError`.

## How to Compile and Run

1. Compile the program using `cmake`:
   ```bash
   cd build && cmake .. && make
   ```
2. Run the compiled program:
   ```bash
   ./vector_add
   ```

## Expected Output

The program prints the following messages:
- "CUDA Vector Addition" at the start.
- "Vector addition completed successfully!" if the computation is correct.

If there are any errors during execution, appropriate error messages will be displayed.

## Grid and Block Configuration

The kernel is launched with:
- `threadsPerBlock = 256`
- `blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock`

This configuration ensures that all elements of the vectors are processed efficiently.

## Notes

- The program assumes `N = 10000` as the size of the vectors. You can modify this value in the source code.
- Ensure that your system has a CUDA-capable GPU and the CUDA toolkit installed to compile and run the program.
