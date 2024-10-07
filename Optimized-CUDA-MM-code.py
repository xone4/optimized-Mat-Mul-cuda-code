import cupy as cp
import numpy as np
import time

# Constants for scaling and performance
MAX_SCALE = 65504.0  # Maximum value representable in float16
WARP_SIZE = 32
BLOCK_SIZE = (32, 32)

# Improved kernel with optimized memory access, warp utilization, and Pharaonic multiplication technique
optimized_kernel = cp.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void optimized_kernel(const half2* __restrict__ A, const half2* __restrict__ B, float* __restrict__ C,
                      int M, int N, int K, float scale, bool is_conv, int stride, int padding) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ half2 As[32][32];
    __shared__ half2 Bs[32][32];

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += 32) {
        // Coalesced memory access for loading data
        if (is_conv) {
            // Logic for convolution operation
            int h_out = row / N;
            int w_out = col % N;
            int h_in = h_out * stride - padding;
            int w_in = w_out * stride - padding;

            As[threadIdx.y][threadIdx.x] = (h_in >= 0 && h_in < M && w_in >= 0 && w_in < N && tile + threadIdx.x < K)
                ? __ldg(&A[(h_in * N + w_in) * (K/2) + (tile/2 + threadIdx.x)])
                : __float2half2_rn(0.0f);
        } else {
            // Standard matrix multiplication
            As[threadIdx.y][threadIdx.x] = (row < M && tile + threadIdx.x < K)
                ? __ldg(&A[row * (K/2) + (tile/2 + threadIdx.x)])
                : __float2half2_rn(0.0f);
        }

        Bs[threadIdx.y][threadIdx.x] = (col < N && tile + threadIdx.y < K)
            ? __ldg(&B[(tile + threadIdx.y) * (N/2) + col/2])
            : __float2half2_rn(0.0f);

        __syncwarp();

        #pragma unroll
        for (int k = 0; k < 32; k++) {
            half2 a = As[threadIdx.y][k];
            half2 b = Bs[k][threadIdx.x];

            // Pharaonic multiplication technique inspired optimization
            half2 prod;
            asm("{
                "  .reg .f16x2 r_a, r_b, r_prod;
                "  mov.b32 r_a, %1;
                "  mov.b32 r_b, %2;
                "  mul.f16x2 r_prod, r_a, r_b;
                "  mov.b32 %0, r_prod;
                "}" : "=r"(prod) : "r"(a), "r"(b));

            sum += __half2float(prod.x) + __half2float(prod.y);
        }

        __syncwarp();
    }

    if (row < M && col < N) {
        sum *= scale;
        // Apply ReLU activation
        sum = fmaxf(sum, 0.0f);

        // Use atomic operation for convolution (gradient accumulation)
        if (is_conv) {
            atomicAdd(&C[row * N + col], sum);
        } else {
            C[row * N + col] = sum;
        }
    }
}
''', 'optimized_kernel')

# Improved dynamic scaling function
def improved_dynamic_scaling(A, B, is_backward=False, matrix_size=None):
    if is_backward:
        # More aggressive scaling for gradients to prevent vanishing/exploding gradients
        max_val = max(cp.abs(A).max(), cp.abs(B).max()) * 10
    else:
        max_val = max(cp.abs(A).max(), cp.abs(B).max())

    # Dynamic scaling based on matrix size
    if matrix_size is not None:
        base_factor = 0.8
        size_factor = 1e-5
        dynamic_factor = max(0.5, min(1.0, base_factor - (size_factor * matrix_size)))
    else:
        dynamic_factor = 1.0

    scale_factor = min(MAX_SCALE / (max_val * dynamic_factor), 1.0)
    return scale_factor

# Main function to perform optimized GPU operation (matrix multiplication or convolution)
def optimized_gpu_operation(A, B, C=None, is_conv=False, stride=1, padding=0, is_backward=False):
    M, K = A.shape
    _, N = B.shape
    matrix_size = max(M, N, K)

    if C is None:
        C = cp.zeros((M, N), dtype=cp.float32)

    # Convert input matrices to half precision for improved performance
    A_half = A.astype(cp.float16)
    B_half = B.astype(cp.float16)

    # Ensure K is even for half2 operations
    if K % 2 != 0:
        A_half = cp.pad(A_half, ((0, 0), (0, 1)), 'constant')
        B_half = cp.pad(B_half, ((0, 1), (0, 0)), 'constant')
        K += 1

    scale = improved_dynamic_scaling(A, B, is_backward, matrix_size)

    grid_size = ((N + BLOCK_SIZE[0] - 1) // BLOCK_SIZE[0], (M + BLOCK_SIZE[1] - 1) // BLOCK_SIZE[1])

    # Launch the optimized CUDA kernel
    optimized_kernel(grid_size, BLOCK_SIZE, (A_half, B_half, C, M, N, K, scale, is_conv, stride, padding))

    return C

# Function to measure performance and energy consumption
def measure_performance_and_energy(func, A, B, iterations=10):
    start_time = time.time()
    start_energy = cp.cuda.runtime.deviceGetPowerUsage(0)

    for _ in range(iterations):
        result = func(A, B)
        cp.cuda.Stream.null.synchronize()

    end_time = time.time()
    end_energy = cp.cuda.runtime.deviceGetPowerUsage(0)

    execution_time = (end_time - start_time) / iterations
    energy_consumption = (end_energy - start_energy) / iterations

    return result, execution_time, energy_consumption

# Test function to evaluate the performance of the optimized algorithm
def test_optimized_version(size):
    print(f"\nTesting optimized version for matrix size {size}x{size}:")

    A = cp.random.rand(size, size, dtype=cp.float32) * 2 - 1
    B = cp.random.rand(size, size, dtype=cp.float32) * 2 - 1

    # Forward pass
    C, exec_time_forward, energy_forward = measure_performance_and_energy(
        lambda x, y: optimized_gpu_operation(x, y), A, B)

    # Backward pass (gradient w.r.t A)
    grad_A, exec_time_grad_A, energy_grad_A = measure_performance_and_energy(
        lambda x, y: optimized_gpu_operation(x, y, is_backward=True), B.T, C.T)
    grad_A = grad_A.T

    # Backward pass (gradient w.r.t B)
    grad_B, exec_time_grad_B, energy_grad_B = measure_performance_and_energy(
        lambda x, y: optimized_gpu_operation(x, y, is_backward=True), A.T, C)

    print(f"Forward pass - Execution time: {exec_time_forward:.6f} s, Energy: {energy_forward:.2f} W")
    print(f"Backward pass (grad A) - Execution time: {exec_time_grad_A:.6f} s, Energy: {energy_grad_A:.2f} W")
    print(f"Backward pass (grad B) - Execution time: {exec_time_grad_B:.6f} s, Energy: {energy_grad_B:.2f} W")

    # Comparison with CuPy for accuracy check
    cupy_C = cp.dot(A, B)
    cupy_grad_A = cp.dot(C, B.T)
    cupy_grad_B = cp.dot(A.T, C)

    print(f"Accuracy difference (forward): {cp.abs(C - cupy_C).max()}")
    print(f"Accuracy difference (grad A): {cp.abs(grad_A - cupy_grad_A).max()}")
    print(f"Accuracy difference (grad B): {cp.abs(grad_B - cupy_grad_B).max()}")

# Main execution
if __name__ == "__main__":
    # Test on different matrix sizes
    sizes = [1024, 2048, 4096]
    for size in sizes:
        test_optimized_version(size)

    # Performance analysis using CUDA profiler
    cp.cuda.profiler.start()
    optimized_gpu_operation(cp.random.rand(2048, 2048, dtype=cp.float32), cp.random.rand(2048, 2048, dtype=cp.float32))
    cp.cuda.profiler.stop()
