# Experiment 6: Memory Optimization in CUDA
### Coalesced vs. Uncoalesced Memory Access | Matrix Transpose | Matrix Multiplication

![CUDA](https://img.shields.io/badge/CUDA-13.4-76B900?logo=nvidia)
![C++](https://img.shields.io/badge/Language-C%2B%2B-00599C?logo=cplusplus)
![GPU](https://img.shields.io/badge/GPU-RTX%202050-76B900?logo=nvidia)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?logo=windows)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 1. Overview

This project implements and evaluates CUDA-based memory optimization techniques using matrix operations. The primary objective is to understand how GPU global memory access patterns influence kernel performance, particularly the difference between **coalesced and uncoalesced memory access**.

The experiment implements and evaluates:

1. **Uncoalesced Matrix Transpose:** A baseline implementation where neighboring threads access memory locations that are not arranged for efficient global memory transactions.
2. **Coalesced Matrix Transpose:** An optimized implementation that uses shared memory to improve global memory access patterns.
3. **Naive Matrix Multiplication:** A basic CUDA implementation of matrix multiplication to evaluate GPU computational throughput.

The implementations are verified against CPU reference calculations, and their execution performance is measured using CUDA events. This demonstrates how appropriate memory access patterns can improve GPU performance without changing the underlying mathematical operation.

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
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 2050 |
| **GPU Architecture** | Ampere |
| **Compute Capability** | 8.6 |
| **GPU Memory** | 4 GB |
| **Operating System** | Windows |
| **CPU** | Host processor used for CPU verification |

### Software Specifications

| Component | Version / Tool |
| :--- | :--- |
| **CUDA Toolkit** | 13.4.92 |
| **CUDA Compiler** | NVCC |
| **NVIDIA Driver** | 592.27 |
| **Programming Language** | CUDA C++ |
| **Build Environment** | Visual Studio x64 Native Tools Command Prompt |
| **Version Control** | Git |

> **Note:** The reported measurements are from the RTX 2050 system used during this experiment. Performance may vary depending on GPU architecture, clock frequency, memory configuration, driver, and system workload.

---

## 4. Concepts Behind the Experiment

### 4.1 GPU Memory Access
A GPU executes thousands of threads in parallel, frequently accessing global memory to retrieve input data and store computed results. The efficiency of these memory operations depends on the access pattern: when neighboring threads access nearby memory addresses, the GPU combines their requests into fewer memory transactions—known as **memory coalescing**.

### 4.2 Coalesced Memory Access
Coalesced access occurs when threads in a warp access consecutive or suitably aligned memory locations:

```text
Thread 0  → A[0]
Thread 1  → A[1]
Thread 2  → A[2]
Thread 3  → A[3]
...
