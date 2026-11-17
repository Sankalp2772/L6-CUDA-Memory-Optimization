
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32
#define BLK 16

// CUDA error checking
#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n", \
                cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// =====================================================
// KERNEL 1: UNCOALESCED MATRIX TRANSPOSE
// =====================================================

__global__ void transposeNaive(
    const float *in, float *out, int N)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < N)
    {
        out[x * N + y] = in[y * N + x];
    }
}

// =====================================================
// KERNEL 2: COALESCED MATRIX TRANSPOSE
// =====================================================

__global__ void transposeCoalesced(
    const float *in, float *out, int N)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < N)
    {
        tile[threadIdx.y][threadIdx.x] =
            in[y * N + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < N && y < N)
    {
        out[y * N + x] =
            tile[threadIdx.x][threadIdx.y];
    }
}

// =====================================================
// KERNEL 3: NAIVE MATRIX MULTIPLICATION
// =====================================================

__global__ void matMulNaive(
    const float *A,
    const float *B,
    float *C,
    int N)
{
    int row = blockIdx.y * BLK + threadIdx.y;
    int col = blockIdx.x * BLK + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] *
                   B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// =====================================================
// HOST LAUNCH WRAPPERS FOR TRANSPOSE
// =====================================================

typedef void (*TransposeLauncher)(
    dim3, dim3, const float*, float*, int);

void launchNaive(
    dim3 grid, dim3 block,
    const float *in, float *out, int N)
{
    transposeNaive<<<grid, block>>>(in, out, N);
}

void launchCoalesced(
    dim3 grid, dim3 block,
    const float *in, float *out, int N)
{
    transposeCoalesced<<<grid, block>>>(in, out, N);
}

// =====================================================
// CUDA EVENT TIMING
// =====================================================

float timeTranspose(
    TransposeLauncher launch,
    dim3 grid,
    dim3 block,
    const float *in,
    float *out,
    int N,
    int reps)
{
    cudaEvent_t start, stop;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Warm-up
    launch(grid, block, in, out, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Start timing
    CHECK(cudaEventRecord(start));

    for (int i = 0; i < reps; i++)
    {
        launch(grid, block, in, out, N);
    }

    CHECK(cudaGetLastError());

    // Stop timing
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float milliseconds;

    CHECK(cudaEventElapsedTime(
        &milliseconds, start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return milliseconds / reps;
}

// =====================================================
// CPU VERIFICATION: TRANSPOSE
// =====================================================

bool checkTranspose(
    const float *in,
    const float *out,
    int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (out[j * N + i] != in[i * N + j])
                return false;
        }
    }

    return true;
}

// =====================================================
// CPU VERIFICATION: MATRIX MULTIPLICATION
// =====================================================

bool checkMatMul(
    const float *A,
    const float *B,
    const float *C,
    int N)
{
    for (int t = 0; t < 20; t++)
    {
        int i = rand() % N;
        int j = rand() % N;

        double ref = 0.0;

        for (int k = 0; k < N; k++)
        {
            ref += (double)A[i * N + k] *
                   B[k * N + j];
        }

        if (fabs(ref - C[i * N + j]) >
            1e-3 * fabs(ref) + 1e-3)
        {
            return false;
        }
    }

    return true;
}

// =====================================================
// MAIN FUNCTION
// =====================================================

int main(int argc, char **argv)
{
    int NT = 4096;
    int NM = 1024;

    if (argc > 1) NT = atoi(argv[1]);
    if (argc > 2) NM = atoi(argv[2]);

    if (NT <= 0 || NM <= 0)
    {
        printf("Invalid matrix dimensions!\n");
        return 1;
    }

    srand(0);

    // GPU information
    cudaDeviceProp prop;

    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("\n========================================\n");
    printf(" CUDA MEMORY OPTIMIZATION EXPERIMENT\n");
    printf("========================================\n");

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n",
           prop.major, prop.minor);

    // =================================================
    // PART A: MATRIX TRANSPOSE
    // =================================================

    printf("\nPART A: MATRIX TRANSPOSE\n");

    size_t bytesT =
        (size_t)NT * NT * sizeof(float);

    float *h_in = (float*)malloc(bytesT);
    float *h_out = (float*)malloc(bytesT);

    if (!h_in || !h_out)
    {
        printf("Host memory allocation failed!\n");
        return 1;
    }

    for (size_t i = 0; i < (size_t)NT * NT; i++)
    {
        h_in[i] = (float)(i % 1000);
    }

    float *d_in, *d_out;

    CHECK(cudaMalloc((void**)&d_in, bytesT));
    CHECK(cudaMalloc((void**)&d_out, bytesT));

    CHECK(cudaMemcpy(
        d_in, h_in, bytesT,
        cudaMemcpyHostToDevice));

    dim3 blockT(TILE, TILE);

    dim3 gridT(
        (NT + TILE - 1) / TILE,
        (NT + TILE - 1) / TILE);

    // Uncoalesced transpose
    float tN = timeTranspose(
        launchNaive, gridT, blockT,
        d_in, d_out, NT, 20);

    CHECK(cudaMemcpy(
        h_out, d_out, bytesT,
        cudaMemcpyDeviceToHost));

    bool okN = checkTranspose(h_in, h_out, NT);

    // Coalesced transpose
    float tC = timeTranspose(
        launchCoalesced, gridT, blockT,
        d_in, d_out, NT, 20);

    CHECK(cudaMemcpy(
        h_out, d_out, bytesT,
        cudaMemcpyDeviceToHost));

    bool okC = checkTranspose(h_in, h_out, NT);

    double gbN = 2.0 * bytesT / (tN * 1e6);
    double gbC = 2.0 * bytesT / (tC * 1e6);

    printf("\nMatrix size: %d x %d\n", NT, NT);

    printf("Uncoalesced : %.3f ms | %.2f GB/s | %s\n",
           tN, gbN, okN ? "PASS" : "FAIL");

    printf("Coalesced   : %.3f ms | %.2f GB/s | %s\n",
           tC, gbC, okC ? "PASS" : "FAIL");

    printf("Speedup     : %.2fx\n", tN / tC);

    // =================================================
    // PART B: NAIVE MATRIX MULTIPLICATION
    // =================================================

    printf("\nPART B: NAIVE MATRIX MULTIPLICATION\n");

    size_t bytesM =
        (size_t)NM * NM * sizeof(float);

    float *hA = (float*)malloc(bytesM);
    float *hB = (float*)malloc(bytesM);
    float *hC = (float*)malloc(bytesM);

    if (!hA || !hB || !hC)
    {
        printf("Host memory allocation failed!\n");
        return 1;
    }

    for (size_t i = 0; i < (size_t)NM * NM; i++)
    {
        hA[i] = rand() / (float)RAND_MAX;
        hB[i] = rand() / (float)RAND_MAX;
    }

    float *dA, *dB, *dC;

    CHECK(cudaMalloc((void**)&dA, bytesM));
    CHECK(cudaMalloc((void**)&dB, bytesM));
    CHECK(cudaMalloc((void**)&dC, bytesM));

    CHECK(cudaMemcpy(
        dA, hA, bytesM, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(
        dB, hB, bytesM, cudaMemcpyHostToDevice));

    dim3 block(BLK, BLK);

    dim3 grid(
        (NM + BLK - 1) / BLK,
        (NM + BLK - 1) / BLK);

    // Warm-up
    matMulNaive<<<grid, block>>>(dA, dB, dC, NM);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // CUDA event timing
    cudaEvent_t start, stop;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    int reps = 5;

    CHECK(cudaEventRecord(start));

    for (int r = 0; r < reps; r++)
    {
        matMulNaive<<<grid, block>>>(dA, dB, dC, NM);
    }

    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float milliseconds;

    CHECK(cudaEventElapsedTime(
        &milliseconds, start, stop));

    float tM = milliseconds / reps;

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    CHECK(cudaMemcpy(
        hC, dC, bytesM,
        cudaMemcpyDeviceToHost));

    bool okM = checkMatMul(hA, hB, hC, NM);

    double gflops =
        2.0 * NM * NM * (double)NM /
        (tM * 1e6);

    printf("Matrix size: %d x %d\n", NM, NM);

    printf("Naive MatMul: %.3f ms | %.2f GFLOPS | %s\n",
           tM, gflops, okM ? "PASS" : "FAIL");

    // =================================================
    // SUMMARY
    // =================================================

    printf("\n========================================\n");
    printf(" EXPERIMENT SUMMARY\n");
    printf("========================================\n");

    printf("Uncoalesced transpose : %.3f ms\n", tN);
    printf("Coalesced transpose   : %.3f ms\n", tC);
    printf("Naive matrix multiply : %.3f ms\n", tM);

    printf("\nVerification:\n");
    printf("Transpose Naive      : %s\n", okN ? "PASS" : "FAIL");
    printf("Transpose Coalesced   : %s\n", okC ? "PASS" : "FAIL");
    printf("Matrix Multiplication : %s\n", okM ? "PASS" : "FAIL");

    printf("\n========================================\n");

    // Free GPU memory
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));

    // Free CPU memory
    free(h_in);
    free(h_out);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
