#include "matmul.h"

void mat_mul(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
  for (size_t m = 0; m < M; m++) 
    for (size_t k = 0; k < K; k++)
      for (size_t n = 0; n < N; n++)
        C[m * N + n] += A[m * K + k] * B[k * N + n]; 
}
