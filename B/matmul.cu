#include "matmul.h"

#include <cuda_runtime.h>

#define ceil(n,m) (((n -  1) % m) + 1)

const int TILE_SIZE = 31;

__global__ void sgemm(float* A, float* B, float* C, size_t M, size_t N, size_t K) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;
  int num_tiles = (K - 1) / TILE_SIZE + 1;
  
  __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
  __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

  float temp = 0;
  for (int tile = 0; tile < num_tiles; tile++) {
    int a_index = row * K + tile * TILE_SIZE + threadIdx.y;
    int b_index = col * K + tile * TILE_SIZE + threadIdx.x;
    
    bool boundary = tile == (num_tiles - 1);
    int k_limit = (boundary ? ceil(K, TILE_SIZE) : TILE_SIZE);

    bool a_cond = (a_index < M * K && threadIdx.y < k_limit);
    A_tile[threadIdx.x][threadIdx.y] = a_cond ? A[a_index] : 0;

    bool b_cond = (b_index < N * K && threadIdx.x < k_limit);
    B_tile[threadIdx.y][threadIdx.x] = b_cond ? B[b_index] : 0;
    
    __syncthreads();

    for (int k = 0; k < k_limit; k++)
      temp += A_tile[threadIdx.x][k] * B_tile[threadIdx.y][k];
    __syncthreads();

  }

  if (row < M && col < N)
    C[row * N + col] = temp;
}

void mat_mul(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
  dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
  dim3 gridDim((M+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE, 1);
  sgemm<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  cudaDeviceSynchronize();
}
