# Experiment 6: Memory Optimization in CUDA
### Coalesced vs. Uncoalesced Memory Access | Matrix Transpose | Matrix Multiplication

![CUDA](https://img.shields.io/badge/CUDA-13.4-76B900?logo=nvidia)
![C++](https://img.shields.io/badge/Language-C%2B%2B-00599C?logo=cplusplus)
![GPU](https://img.shields.io/badge/GPU-RTX%202050-76B900?logo=nvidia)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?logo=windows)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 1. Overview

This project implements and evaluates CUDA-based memory optimization techniques using matrix operations.

The primary objective is to understand how GPU global memory access patterns influence kernel performance, particularly the difference between **coalesced and uncoalesced memory access**.

The experiment implements and evaluates:

1. **Uncoalesced Matrix Transpose:** A baseline implementation where neighboring threads access memory locations that are not arranged for efficient global memory transactions.
2. **Coalesced Matrix Transpose:** An optimized implementation that uses shared memory to improve global memory access patterns.
3. **Naive Matrix Multiplication:** A basic CUDA implementation of matrix multiplication to evaluate GPU computational throughput.

The implementations are verified against CPU reference calculations, and their execution performance is measured using CUDA events.

The experiment demonstrates how appropriate memory access patterns can improve GPU performance without changing the underlying mathematical operation.

---

## 2. Objectives

- Understand GPU global memory access and memory coalescing.
- Implement matrix transpose using uncoalesced memory access.
- Optimize matrix transpose using coalesced memory access.
- Understand the role of shared memory in CUDA optimization.
- Implement naive matrix multiplication using CUDA.
- Measure kernel execution time using CUDA events.
- Calculate effective memory bandwidth and floating-point throughput.
- Compare optimized and baseline implementations.
- Verify GPU results against CPU reference implementations.

---

## 3. Experimental Environment

### Hardware Specifications

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 2050 |
| GPU Architecture | Ampere |
| Compute Capability | 8.6 |
| GPU Memory | 4 GB |
| Operating System | Windows |
| CPU | Host processor used for CPU verification |

### Software Specifications

| Component | Version / Tool |
|---|---|
| CUDA Toolkit | 13.4.92 |
| CUDA Compiler | NVCC |
| NVIDIA Driver | 592.27 |
| Programming Language | CUDA C++ |
| Build Environment | Visual Studio x64 Native Tools Command Prompt |
| Version Control | Git |

**Note:** The reported measurements are from the RTX 2050 system used during this experiment. Performance may vary depending on GPU architecture, clock frequency, memory configuration, driver, and system workload.

---

## 4. Concepts Behind the Experiment

### 4.1 GPU Memory Access

A GPU executes thousands of threads in parallel. Threads frequently access global memory to retrieve input data and store computed results.

The efficiency of these memory operations depends on the memory access pattern.

When neighboring threads access nearby memory addresses, the GPU can combine their requests into fewer memory transactions.

This is known as **memory coalescing**.

### 4.2 Coalesced Memory Access

Coalesced access occurs when threads in a warp access consecutive or suitably aligned memory locations.

For example, when threads access consecutive elements:

```text
Thread 0  → A[0]
Thread 1  → A[1]
Thread 2  → A[2]
Thread 3  → A[3]
...
```

The memory requests can be serviced efficiently through fewer transactions.

**Advantages:**
- Better global memory utilization.
- Reduced memory transaction overhead.
- Improved effective memory bandwidth.
- Better kernel performance for memory-bound operations.

### 4.3 Uncoalesced Memory Access

Uncoalesced access occurs when neighboring threads access memory locations that are far apart or poorly arranged.

For example:

```text
Thread 0  → A[0][0]
Thread 1  → A[1][0]
Thread 2  → A[2][0]
Thread 3  → A[3][0]
...
```

For a row-major matrix, these addresses are separated by an entire row stride.

This can result in inefficient global memory transactions and lower effective memory bandwidth.

### 4.4 Role of Shared Memory

CUDA shared memory is a fast, on-chip memory space shared by threads within a thread block.

In the optimized transpose implementation:

1. Threads read matrix elements from global memory in a coalesced pattern.
2. Elements are stored in a shared-memory tile.
3. Threads synchronize using `__syncthreads()`.
4. Threads read the transposed tile from shared memory.
5. Results are written back to global memory in a coalesced pattern.

This allows the implementation to transform an inefficient global memory access pattern into an efficient one.

### 4.5 Shared Memory Padding

The optimized transpose uses a shared-memory tile with dimensions:

```cpp
__shared__ float tile[32][33];
```

Instead of using a 32 × 32 tile, an additional column is introduced.

This padding helps avoid shared-memory bank conflicts during transposed accesses.

---

## 5. Implementation Details

### Part A: Matrix Transpose

Matrix transpose converts rows into columns and columns into rows.

For a matrix A:

\[
A_{M \times N}
\]

The transposed matrix is:

\[
B_{N \times M}
\]

The mathematical operation is:

\[
B[j][i] = A[i][j]
\]

#### A. Uncoalesced Matrix Transpose

The baseline implementation performs matrix transpose directly using global memory.

Conceptually:

```cpp
output[col * N + row] = input[row * N + col];
```

When neighboring threads vary their row index while keeping the column index fixed, their global memory accesses are separated by the matrix stride.

This produces an inefficient access pattern for row-major storage.

#### B. Coalesced Matrix Transpose

The optimized implementation uses shared memory to rearrange matrix elements before writing the transposed output.

Conceptual implementation:

```cpp
__shared__ float tile[32][33];

// Read input matrix into shared memory
tile[threadIdx.y][threadIdx.x] =
    input[row * N + col];

__syncthreads();

// Write transposed elements
output[newRow * N + newCol] =
    tile[threadIdx.x][threadIdx.y];
```

The actual implementation also needs appropriate boundary checks and thread/block indexing.

**Optimization achieved:** Global memory reads and writes are organized to improve coalescing.

### Part B: Naive Matrix Multiplication

Matrix multiplication is implemented using CUDA threads.

Given two matrices:

\[
A_{M \times K}, \quad B_{K \times N}
\]

The output matrix is:

\[
C_{M \times N}
\]

Each output element is calculated as:

\[
C[i][j] = \sum_{k=0}^{K-1} A[i][k] \times B[k][j]
\]

Each CUDA thread computes one output element.

Conceptual implementation:

```cpp
float sum = 0.0f;

for (int k = 0; k < K; k++) {
    sum += A[row * K + k] *
           B[k * N + col];
}

C[row * N + col] = sum;
```

This is a naive implementation without shared-memory tiling or advanced matrix multiplication optimizations.

---

## 6. Performance Measurement

CUDA events are used to measure GPU kernel execution time.

The general procedure is:

1. Create CUDA start and stop events.
2. Record the start event.
3. Launch the CUDA kernel.
4. Record the stop event.
5. Synchronize the stop event.
6. Calculate elapsed time.

Example:

```cpp
cudaEventRecord(start);

kernel<<<grid, block>>>(...);

cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0.0f;

cudaEventElapsedTime(
    &milliseconds,
    start,
    stop
);
```

### Performance Metrics

**Execution Time**

The time taken by a CUDA kernel to complete its computation.

**Effective Memory Bandwidth**

For matrix transpose, the approximate effective bandwidth is calculated using:

\[
Bandwidth = \frac{2 \times N \times N \times sizeof(float)}
{Execution\ Time}
\]

The factor of 2 accounts for reading the input matrix and writing the output matrix.

The result is converted into GB/s using consistent units.

**Floating-Point Operations Per Second**

For matrix multiplication, the approximate number of floating-point operations is:

\[
FLOPs = 2 \times M \times N \times K
\]

The throughput is:

\[
GFLOPS = \frac{FLOPs}{Execution\ Time \times 10^9}
\]

Execution time must be expressed in seconds for these calculations.

**Speedup**

The speedup of the optimized transpose relative to the baseline is:

\[
Speedup =
\frac{T_{uncoalesced}}{T_{coalesced}}
\]

A speedup greater than 1 indicates that the optimized implementation is faster.

---

## 7. Experimental Results

The experiment was executed using:

```cmd
exp6.exe 1024 256
```

### A. Matrix Transpose Results

Matrix dimensions: **1024 × 1024**

| Metric | Uncoalesced | Coalesced |
|---|---:|---:|
| Execution Time | 0.220 ms | 0.113 ms |
| Effective Bandwidth | 38.11 GB/s | 74.15 GB/s |
| Verification | PASS | PASS |

### B. Matrix Multiplication Results

Matrix dimensions: **256 × 256**

| Metric | Result |
|---|---:|
| Implementation | Naive CUDA Matrix Multiplication |
| Execution Time | 0.098 ms |
| Performance | 343.48 GFLOPS |
| Verification | PASS |

### C. Performance Summary

| Operation | Execution Time | Result |
|---|---:|---|
| Uncoalesced Transpose | 0.220 ms | PASS |
| Coalesced Transpose | 0.113 ms | PASS |
| Naive Matrix Multiplication | 0.098 ms | PASS |

### D. Transpose Speedup

\[
Speedup = \frac{0.220}{0.113}
\]

\[
\boxed{Speedup \approx 1.95\times}
\]

The coalesced transpose is approximately **1.95 times faster** than the uncoalesced implementation.

Execution time reduction:

\[
\frac{0.220-0.113}{0.220}\times100
\]

\[
\boxed{48.64\%}
\]

Effective bandwidth improvement:

\[
\frac{74.15-38.11}{38.11}\times100
\]

\[
\boxed{94.57\%}
\]

The optimized transpose achieves approximately 94.57% higher effective bandwidth in this measured run.

---

## 8. Performance Analysis

The experimental results demonstrate the importance of memory access patterns in GPU programming.

### Observation 1: Coalesced Access Improves Performance

The uncoalesced transpose takes 0.220 ms, whereas the coalesced transpose takes only 0.113 ms.

The improvement occurs because neighboring threads access global memory more efficiently in the optimized implementation.

### Observation 2: Higher Effective Bandwidth

The effective bandwidth increases from 38.11 GB/s to 74.15 GB/s.

This indicates improved utilization of global memory bandwidth.

### Observation 3: Shared Memory Helps Reorganize Data

Shared memory enables threads to rearrange matrix elements before writing the transposed matrix.

This reduces inefficient global memory access patterns.

### Observation 4: Matrix Multiplication Performance

The naive matrix multiplication achieves 343.48 GFLOPS for a 256 × 256 matrix.

Although this implementation performs the required operation successfully, it does not use advanced techniques such as shared-memory tiling, register blocking, or Tensor Cores.

### Overall Analysis

The experiment establishes that GPU performance depends not only on the number of threads executing in parallel but also on how those threads access memory.

A well-designed memory access pattern can significantly improve performance without changing the mathematical operation.

---

## 9. Correctness Verification

All three CUDA implementations were checked against CPU reference calculations.

| Test Case | Verification |
|---|---|
| Uncoalesced Matrix Transpose | PASS |
| Coalesced Matrix Transpose | PASS |
| Naive Matrix Multiplication | PASS |

The output confirms that the GPU implementations produce results matching the CPU reference within the verification tolerance used by the program.

This validates the correctness of the implementations for the tested matrix dimensions.

---

## 10. How to Build and Execute

### Prerequisites

- NVIDIA GPU with CUDA support.
- NVIDIA GPU driver.
- CUDA Toolkit.
- Visual Studio C++ Build Tools.
- Git (optional).

### Step 1: Clone the Repository

```cmd
git clone https://github.com/Sankalp2772/L6-CUDA-Memory-Optimization.git
```

### Step 2: Navigate to the Project

```cmd
cd L6-CUDA-Memory-Optimization
```

### Step 3: Open the x64 Native Tools Command Prompt

Use the Visual Studio x64 Native Tools Command Prompt to ensure the required Microsoft C++ compiler tools are available.

### Step 4: Compile the CUDA Program

```cmd
nvcc -O2 -gencode=arch=compute_86,code=sm_86 exp6.cu -o exp6.exe
```

The compilation command generates native machine code for Compute Capability 8.6.

### Step 5: Execute

```cmd
exp6.exe
```

To specify matrix dimensions:

```cmd
exp6.exe 1024 256
```

Here:

- `1024` specifies the matrix dimension used for matrix transpose.
- `256` specifies the matrix dimension used for matrix multiplication.

### Step 6: Verify the Output

The program reports:
- Detected GPU.
- Compute capability.
- Matrix transpose timings.
- Effective memory bandwidth.
- Transpose speedup.
- Matrix multiplication execution time.
- GFLOPS.
- CPU verification results.

---

## 11. Expected Output

The following is the measured output from the experimental run:

```text
CUDA MEMORY OPTIMIZATION EXPERIMENT
GPU: NVIDIA GeForce RTX 2050
Compute Capability: 8.6

PART A: MATRIX TRANSPOSE
Matrix size: 1024 x 1024
Uncoalesced : 0.220 ms | 38.11 GB/s | PASS
Coalesced   : 0.113 ms | 74.15 GB/s | PASS
Speedup     : 1.95x

PART B: NAIVE MATRIX MULTIPLICATION
Matrix size: 256 x 256
Naive MatMul: 0.098 ms | 343.48 GFLOPS | PASS

EXPERIMENT SUMMARY
Uncoalesced transpose : 0.220 ms
Coalesced transpose   : 0.113 ms
Naive matrix multiply : 0.098 ms

Verification:
Transpose Naive      : PASS
Transpose Coalesced   : PASS
Matrix Multiplication : PASS
```

*Note: Actual timings can differ between executions and GPU systems.*

---

## 12. Key Learnings

Through this experiment, the following concepts were explored:

1. Understanding CUDA global memory access patterns.
2. Difference between coalesced and uncoalesced memory access.
3. Importance of memory bandwidth in GPU performance.
4. Usage of shared memory for data rearrangement.
5. Shared-memory bank conflicts and padding.
6. CUDA thread and block execution.
7. CUDA event-based performance measurement.
8. CPU-based correctness verification.
9. Calculation of effective bandwidth and GFLOPS.
10. Performance comparison between baseline and optimized CUDA kernels.

---

## 13. Limitations and Future Improvements

The experiment uses basic CUDA implementations to demonstrate memory optimization concepts.

Potential improvements include:

- Implement tiled matrix multiplication using shared memory.
- Compare naive and tiled matrix multiplication.
- Explore different transpose tile dimensions.
- Evaluate the impact of matrix size on performance.
- Perform repeated measurements and calculate average execution time.
- Investigate CUDA occupancy and memory throughput using NVIDIA Nsight Compute.
- Compare performance across different GPU architectures.
- Explore asynchronous memory operations and advanced CUDA optimization techniques.

---

## 14. Conclusion

This experiment demonstrates the impact of memory access patterns on CUDA kernel performance.

The comparison between uncoalesced and coalesced matrix transpose shows that optimizing global memory access can substantially improve performance.

On the NVIDIA GeForce RTX 2050, the coalesced transpose achieved approximately **1.95× speedup**, reducing execution time from 0.220 ms to 0.113 ms.

The effective memory bandwidth also increased from 38.11 GB/s to 74.15 GB/s.

The naive matrix multiplication achieved 343.48 GFLOPS for a 256 × 256 matrix.

All three implementations passed CPU-based correctness verification.

Overall, the experiment highlights that **efficient memory access is a fundamental factor in achieving high performance in GPU computing**.

---

## 15. Repository Structure

```text
L6-CUDA-Memory-Optimization/
│
├── exp6.cu
├── README.md
├── .gitignore
│
├── results/
│   └── experiment-output.txt
│
└── screenshots/
    └── execution-results.png
```

*The `results` and `screenshots` directories are suggested locations for storing experimental output and execution evidence.*

---

## Author

**Sankalp Patil**

B.E. Computer Science and Artificial Intelligence  
KLE Technological University, Hubballi

**Experiment:** 06 – Memory Optimization in CUDA

**Repository:** [L6-CUDA-Memory-Optimization](https://github.com/Sankalp2772/L6-CUDA-Memory-Optimization)

---

⭐ If you find this experiment useful for understanding CUDA memory optimization, feel free to explore the implementation and experiment with different matrix dimensions.
