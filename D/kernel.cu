#include "util.h"
#include "matmul.h"
#include "kernel.h"
#include <string>
#include <map>
#include <cmath>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <stdio.h>
#include <cuda_runtime.h>

#define ceil(n,m) (((n -  1) % m) + 1)

const size_t GPUS = 4;
const static size_t BATCH = 4;

class Tensor {
public:
  Tensor();
  Tensor(float *buf_, std::vector<size_t> shape_);
  void alloc_once(std::vector<size_t> shape_);
  void alloc_on_host(std::vector<size_t> shape_);
  void set_sz();
  void set_cuda_buf();
  void init_cuda_buf();
  void set_buf_from_cuda();
  void free_cuda_buf();
  void set_cuda_buf(float *b);

  // For real world application, one should use smart pointer to prevent possible memory leak.
  // However, we just use raw pointer for simplicity. Memory will be freed anyway when process exits.
  float* buf;
  float* cuda_buf;
  bool gpu = false;

  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., [[1, 2, 3], [4, 5, 6]] => shape = [2, 3]
  std::vector<size_t> shape;

  // Size of tensor; product of all dimensions
  size_t sz;
};

// Helpers
static void register_weight(std::map<std::string, Tensor>& weights, float* (*buf), std::string name, std::vector<size_t> shape);
static std::map<std::string, Tensor> register_weights(float* weight_buf);
static Tensor preprocess(uint8_t *in, size_t num_image);
static void postprocess_one_image(Tensor input, uint8_t *out, size_t idx);
static void get_one_image(Tensor input, Tensor &output, size_t idx);

// Operators
static void conv2d(Tensor input, Tensor filter, Tensor bias, Tensor &output, Tensor &reshaped_input);
static void conv2d_transposed(Tensor input, Tensor filter, Tensor bias, Tensor &output, Tensor &reshaped_input);
static void leaky_relu(Tensor input, Tensor &output, float alpha);
static void relu(Tensor input, Tensor &output);
static void batchnorm(Tensor input, Tensor scale, Tensor offset, Tensor &output);
static void concat(Tensor input0, Tensor input1, Tensor &output);
static void elem_tanh(Tensor input, Tensor &output);
static void reshape_filter(Tensor *input);

static void matmul(Tensor A, Tensor B, Tensor &C, size_t M, size_t N, size_t K);

// Profilers
static double conv2d_t = 0, conv2d_tr_t = 0, leaky_relu_t = 0, relu_t = 0, batchnorm_t = 0, concat_t = 0, tanh_t = 0;
static double im2col_t = 0, shift_t = 0, reshape_t = 0, matmul_t = 0;

void _pix2pix(uint8_t *input_buf, float *weight_buf, uint8_t *output_buf, size_t num_image) {
  /*
   * !!!!!!!! Caution !!!!!!!!
   * In MPI program, all buffers and num_image are only given to rank 0 process.
   * You should manually:
   *   1. allocate buffers on others
   *   2. send inputs from rank 0 to others
   *   3. gather outputs from others to rank 0
   */
  
  std::map<std::string, Tensor> weights[GPUS];
  for (size_t GPU = 0; GPU < GPUS; GPU++) {
    cudaSetDevice(GPU);
    weights[GPU] = register_weights(weight_buf); // Memory allocated for weights
  }
  auto input = preprocess(input_buf, num_image); // Memory allocated for input

  printf("Weights & Preprocess Done\n");

  // Declare feature maps
  // Memory for feature maps are allocated when they are written first time using Tensor::alloc_once(...)
  Tensor one_image[GPUS][BATCH];
  Tensor encoder_layer_input[GPUS][BATCH][9];
  Tensor encoder_layer_rectified[GPUS][BATCH][9];
  Tensor encoder_layer_reshaped[GPUS][BATCH][9];
  Tensor encoder_layer_convolved[GPUS][BATCH][9];
  Tensor encoder_layer[GPUS][BATCH][9];
  Tensor decoder_layer_input[GPUS][BATCH][9];
  Tensor decoder_layer_rectified[GPUS][BATCH][9];
  Tensor decoder_layer_reshaped[GPUS][BATCH][9];
  Tensor decoder_layer_convolved[GPUS][BATCH][9];
  Tensor decoder_layer[GPUS][BATCH][9];
  
  // reshape filters
  for (int i = 8; i >= 1; --i) {
    for (size_t GPU = 0; GPU < GPUS; GPU++) {
      cudaSetDevice(GPU);
      auto decoder = &weights[GPU]["generator/decoder_" + std::to_string(i) + "/conv2d_transpose/kernel"];
      reshape_filter(decoder);
    }
  }
  
  size_t curr_batch[GPUS];

  for (size_t img_idx = 0; img_idx < num_image; img_idx += GPUS * BATCH) {
    size_t total_images = (num_image - img_idx) < GPUS * BATCH ? (num_image - img_idx) : GPUS * BATCH;
    size_t curr_images = total_images >= 4 ? 4 : total_images;
    for (size_t GPU = 0; GPU < curr_images; GPU++) {
      curr_batch[GPU] = (total_images / GPUS) + ((total_images % GPUS > GPU) ? 1 : 0);
    }
    
    // Pick 1 image out of num_image
    int curr_img_idx = img_idx;
    for (size_t GPU = 0; GPU < curr_images; GPU++) {
      cudaSetDevice(GPU);
      for (size_t batch = 0; batch < curr_batch[GPU]; batch++)
        get_one_image(input, one_image[GPU][batch], curr_img_idx++);
    }

    /*
     * Encoding phase
     */

    // Encoder 1 : conv

    for (size_t GPU = 0; GPU < curr_images; GPU++) {
      cudaSetDevice(GPU);
      auto filter = weights[GPU]["generator/encoder_1/conv2d/kernel"];
      auto bias = weights[GPU]["generator/encoder_1/conv2d/bias"];
      for (size_t batch = 0; batch < curr_batch[GPU]; batch++)
          conv2d(one_image[GPU][batch], filter, bias, encoder_layer[GPU][batch][1], encoder_layer_reshaped[GPU][batch][1]);
    }

    for (int i = 2; i <= 8; ++i) {
      // Encoder i : leaky_relu => conv2d => batchnorm
      for (size_t GPU = 0; GPU < curr_images; GPU++) {
        cudaSetDevice(GPU);
        auto scope = "generator/encoder_" + std::to_string(i);
        auto filter = weights[GPU][scope + "/conv2d/kernel"];
        auto bias = weights[GPU][scope + "/conv2d/bias"];
        auto scale = weights[GPU][scope + "/batch_normalization/gamma"];
        auto offset = weights[GPU][scope + "/batch_normalization/beta"];
        for (size_t batch = 0; batch < curr_batch[GPU]; batch++) {
          encoder_layer_input[GPU][batch][i] = encoder_layer[GPU][batch][i - 1];
          leaky_relu(encoder_layer_input[GPU][batch][i], encoder_layer_rectified[GPU][batch][i], 0.2);
          conv2d(encoder_layer_rectified[GPU][batch][i], filter, bias, encoder_layer_convolved[GPU][batch][i], encoder_layer_reshaped[GPU][batch][i]);
          batchnorm(encoder_layer_convolved[GPU][batch][i], scale, offset, encoder_layer[GPU][batch][i]);
        }
      }
    }


    /*
     * Decoding phase
     */

    for (int i = 8; i >= 1; --i) {
      // Decoder i : relu => conv2d_transposed => batchnorm
      for (size_t GPU = 0; GPU < curr_images; GPU++) {
        cudaSetDevice(GPU);
        auto scope = "generator/decoder_" + std::to_string(i);
        auto filter = weights[GPU][scope + "/conv2d_transpose/kernel"];
        auto bias = weights[GPU][scope + "/conv2d_transpose/bias"];
        auto scale = weights[GPU][scope + "/batch_normalization/gamma"];
        auto offset = weights[GPU][scope + "/batch_normalization/beta"];
        for (size_t batch = 0; batch < curr_batch[GPU]; batch++) {
          if (i == 8) {
            // For decoder 8, input is last layer of encoder
            decoder_layer_input[GPU][batch][i] = encoder_layer[GPU][batch][8];
          } else {
            // For other decoder, input is concatenation of previous layer and corresponding encoder layer
            concat(decoder_layer[GPU][batch][i + 1], encoder_layer[GPU][batch][i], decoder_layer_input[GPU][batch][i]);
          }
          relu(decoder_layer_input[GPU][batch][i], decoder_layer_rectified[GPU][batch][i]);
          conv2d_transposed(decoder_layer_rectified[GPU][batch][i], filter, bias, decoder_layer_convolved[GPU][batch][i], decoder_layer_reshaped[GPU][batch][i]);

          // Last decoder does not have batchnorm
          if (i == 1) continue;
          batchnorm(decoder_layer_convolved[GPU][batch][i], scale, offset, decoder_layer[GPU][batch][i]);
        }
      }
    }


    // Convert values into [-1, 1] using tanh function
    for (size_t GPU = 0; GPU < curr_images; GPU++) {
      cudaSetDevice(GPU);
      for (size_t batch = 0; batch < curr_batch[GPU]; batch++)
        elem_tanh(decoder_layer_convolved[GPU][batch][1], decoder_layer[GPU][batch][1]);
    }

    /*
    void postprocess_one_image(Tensor input, uint8_t *out, size_t idx) {
    // input shape = (height, width, channels)
    input.set_buf_from_cuda();
    cudaDeviceSynchronize();
    size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
    for (size_t i = 0; i < H * W * C; ++i) {
    float x = (input.buf[i] + 1) / 2 * 255;
    out[idx * (H * W * C) + i] = x < 0 ? 0 : (x > 255 ? 255 : x);
  }
}
    */

    // Put a image into output buffer
    curr_img_idx = img_idx;
    for (size_t GPU = 0; GPU < curr_images; GPU++) {
      cudaSetDevice(GPU);
      for (size_t batch = 0; batch < curr_batch[GPU]; batch++) {
        decoder_layer[GPU][batch][1].set_buf_from_cuda();
      }
      cudaDeviceSynchronize();
      for (size_t batch = 0; batch < curr_batch[GPU]; batch++) {
        Tensor input = decoder_layer[GPU][batch][1];
        size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
        size_t idx = curr_img_idx++;
        for (size_t i = 0; i < H * W * C; ++i) {
          float x = (input.buf[i] + 1) / 2 * 255;
          output_buf[idx * (H * W * C) + i] = x < 0 ? 0 : (x > 255 ? 255 : x);
        }
      }
    }
  }
  printf("Conv2D: %f\n", conv2d_t);
  printf("Conv2D Transpose: %f\n", conv2d_tr_t);
  printf("LeakyRelu: %f\n", leaky_relu_t);
  printf("Relu: %f\n", relu_t);
  printf("BatchNorm: %f\n", batchnorm_t);
  printf("Concat: %f\n", concat_t);
  printf("Tanh: %f\n", tanh_t);
  
  printf("========\n");
  printf("Im2col: %f\n", im2col_t);
  printf("Shift: %f\n", shift_t);
  printf("Reshape: %f\n", reshape_t);
  printf("Matmul: %f\n", matmul_t);
}

Tensor::Tensor() : buf(NULL){}

// If buf is given, use it. If not, allocate new one.
Tensor::Tensor(float *buf_, std::vector<size_t> shape_) : buf(buf_), shape(shape_) {
  set_sz();
  if (buf == NULL) {
    buf = (float*)malloc(sz * sizeof(float));
  }
  set_cuda_buf();
}

// If buf is not allocated, allocate new one.
void Tensor::alloc_once(std::vector<size_t> shape_) {
  if (!gpu) {
    shape = shape_;
    set_sz();
    init_cuda_buf();
  }
}

void Tensor::alloc_on_host(std::vector<size_t> shape_) {
  if (buf == NULL) {
    shape = shape_;
    set_sz();
    buf = (float*)malloc(sz * sizeof(float));
  }
}

void Tensor::set_sz() {
  sz = 1;
  for (auto x : shape) {
    sz *= x;
  }
}

void Tensor::set_cuda_buf() {
  init_cuda_buf();
  cudaError_t r = cudaMemcpyAsync(cuda_buf, buf, sz * sizeof(float), cudaMemcpyHostToDevice);
  if (r != cudaSuccess) {
    printf(cudaGetErrorString(r));
    printf(", SetCudaBuf error, %d, size: %d\n", r, sz*sizeof(float));
  }// else printf("SetCudaBuf OK, size: %d\n", sz*sizeof(float));
}

void Tensor::set_cuda_buf(float *b) {
  cuda_buf = b;
}

void Tensor::init_cuda_buf() {
  if (gpu) return;
  gpu = true;
  cudaError_t r = cudaMalloc(&cuda_buf, sz * sizeof(float));
  if (r != cudaSuccess){
    printf(cudaGetErrorString(r));
    printf(" CudaMalloc error, size: %d\n" , sz * sizeof(float));
  }
}

void Tensor::free_cuda_buf() {
  cudaError_t r = cudaFree(cuda_buf);
  if (r != cudaSuccess) printf("CudaFree error, %d, size: %d\n", r, sz*sizeof(float));
  gpu = false;
}

void Tensor::set_buf_from_cuda() {
  if (buf == NULL) buf = (float*)malloc(sz * sizeof(float));
  cudaError_t r = cudaMemcpyAsync(buf, cuda_buf, sz * sizeof(float), cudaMemcpyDeviceToHost);
  if (r != cudaSuccess){
    printf(cudaGetErrorString(r));
    printf(", SetBufFromCuda Fail, %d, size: %d\n", r, sz*sizeof(float));
  }
}

// Make a new tensor from buffer and put the tensor into map. Advance buffer pointer by size.
void register_weight(std::map<std::string, Tensor>& weights, float* (*buf), std::string name, std::vector<size_t> shape) {
  Tensor tensor(*buf, shape);
  weights[name] = tensor;
  *buf += tensor.sz;
}

// Put all predefined weights into map. Order should not be changed.
std::map<std::string, Tensor> register_weights(float* weight_buf) {
  std::map<std::string, Tensor> weights;
  // auto generated
  register_weight(weights, &weight_buf, "generator/decoder_1/conv2d_transpose/bias", {3});
  register_weight(weights, &weight_buf, "generator/decoder_1/conv2d_transpose/kernel", {4, 4, 3, 128});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/beta", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/gamma", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/moving_mean", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/moving_variance", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/conv2d_transpose/bias", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/conv2d_transpose/kernel", {4, 4, 64, 256});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/beta", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/gamma", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/moving_mean", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/moving_variance", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/conv2d_transpose/bias", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/conv2d_transpose/kernel", {4, 4, 128, 512});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/beta", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/gamma", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/moving_mean", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/moving_variance", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/conv2d_transpose/bias", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/conv2d_transpose/kernel", {4, 4, 256, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/conv2d_transpose/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_1/conv2d/bias", {64});
  register_weight(weights, &weight_buf, "generator/encoder_1/conv2d/kernel", {4, 4, 3, 64});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/beta", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/gamma", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/moving_mean", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/moving_variance", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/conv2d/bias", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/conv2d/kernel", {4, 4, 64, 128});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/beta", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/gamma", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/moving_mean", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/moving_variance", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/conv2d/bias", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/conv2d/kernel", {4, 4, 128, 256});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/conv2d/kernel", {4, 4, 256, 512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/conv2d/kernel", {4, 4, 512, 512});
  return weights;
}

// Convert 8-bit depth images (value range [0, 255]) into floating-point ones (value range [-1, 1])
Tensor preprocess(uint8_t *in, size_t num_image) {
  Tensor out(NULL, {num_image, 256, 256, 3});
  for (size_t i = 0; i < out.sz; ++i) {
    out.buf[i] = in[i] / 255.0f * 2 - 1;
  }
  return out;
}

// Inverse of preprocess
void postprocess_one_image(Tensor input, uint8_t *out, size_t idx) {
  // input shape = (height, width, channels)
  input.set_buf_from_cuda();
  cudaDeviceSynchronize();
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  for (size_t i = 0; i < H * W * C; ++i) {
    float x = (input.buf[i] + 1) / 2 * 255;
    out[idx * (H * W * C) + i] = x < 0 ? 0 : (x > 255 ? 255 : x);
  }
}

// Pick single image from images
void get_one_image(Tensor input, Tensor &output, size_t idx) {
  // input shape = (num_image, height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[1], W = input.shape[2], C = input.shape[3];
  output.alloc_on_host({H, W, C});
  for (size_t i = 0; i < H * W * C; ++i) {
    output.buf[i] = input.buf[idx * H * W * C + i];
  }
  output.set_cuda_buf();
} 

void matmul(Tensor A, Tensor B, Tensor &C, size_t M, size_t N, size_t K) {
  // printf("MATMUL: M: %d, N: %d, K: %d\n", M, N, K);
  double start = get_time();

  assert(A.sz == (M * K));
  assert(B.sz == (K * N));
  assert(C.sz == (M * N));

  mat_mul(A.cuda_buf, B.cuda_buf, C.cuda_buf, M, N, K);
  matmul_t += (get_time() - start);
}

__global__ void _im2col(float* input, float* output, int OH, int OW, int R, int S, int C) {
  int o = blockDim.x * blockIdx.x + threadIdx.x;
  int rs = blockDim.y * blockIdx.y + threadIdx.y;
  int H = OH * 2 , W = OW * 2;
  int oh = o / OW, ow = o % OW;
  int r = rs / S, s = rs % S;
  int ih = oh * 2 - 1 + r, iw = ow * 2 - 1 + s;
  if (ih < 0 || ih >= H || iw < 0 || iw >= W) return;
  for (int c = 0; c < C; c++)
    output[(oh * OW + ow) * (R * S * C) + (r * S * C + s * C + c)] = input[ih * W * C + iw * C + c];
}

// stride = 2, pad = 1
void im2col(Tensor input, size_t filter_height, size_t filter_width, Tensor &output) {
  double start = get_time();
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t OH = H / 2, OW = W / 2, R = filter_height, S = filter_width;
  output.alloc_once({OH, OW, R, S, C});

  dim3 blockDim(1, 1, 1);
  dim3 gridDim(OH * OW, R * S, 1);
  _im2col<<<gridDim, blockDim>>>(input.cuda_buf, output.cuda_buf, OH, OW, R, S, C);
  im2col_t += (get_time() - start);
}

__global__ void _shift(float* input, float* bias) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  input[row] = bias[threadIdx.x];
}

// shift bias
void shift(Tensor &input, Tensor bias) {
  double start = get_time();
  assert(bias.shape[0] == bias.sz);
  size_t K = bias.sz;
  _shift<<<input.sz / K, K>>>(input.cuda_buf, bias.cuda_buf);
  shift_t += (get_time() - start);
}

// Convolution (2-dimension, stride = 2, pad = 1)
// filter_height = filter_width = 4
void conv2d(Tensor input, Tensor filter, Tensor bias, Tensor &output, Tensor &reshaped_input) {
  double start = get_time();
  // input shape = (in_height, in_width, in_channels)
  // filter shape = (filter_height, filter_width, in_channels, output_channels)
  // bias shape = (output_channels)
  // output shape = (in_height / stride, in_width / stride, output_channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[3];
  const size_t stride = 2, pad = 1;
  size_t OH = H / stride, OW = W / stride;
  output.alloc_once({OH, OW, K});
  im2col(input, R, S, reshaped_input);
  shift(output, bias);
  matmul(reshaped_input, filter, output, OH * OW, K, R * S * C);
  conv2d_t += (get_time() - start);
}

__global__ void _im2col_tr(float* input, float* output, int OH, int OW, int R, int S, int C) {
  int o = blockDim.x * blockIdx.x + threadIdx.x;
  int rs = blockDim.y * blockIdx.y + threadIdx.y;
  int oh = o / OW, ow = o % OW;
  int H = OW / 2, W = OW / 2;
  int r = rs / S, s = rs % S;
  int raw_h = (oh - r + 1), raw_w = (ow - s + 1);
  if (raw_h % 2 != 0 || raw_w % 2 != 0) return;
  int ih = raw_h / 2, iw = raw_w / 2;
  if (ih < 0 || ih >= H || iw < 0 || iw >= W) return;
  for (int c = 0; c < C; c++)
    output[(oh * OW + ow) * (R * S * C) + (r * S * C + s * C + c)] = input[ih * W * C + iw * C + c];
}

void im2col_tr(Tensor input, size_t filter_height, size_t filter_width, Tensor &output) {
  double start = get_time();
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t OH = H * 2, OW = W * 2, R = filter_height, S = filter_width;
  output.alloc_once({OH, OW, R, S, C});
  dim3 blockDim(1, 1, 1);
  dim3 gridDim(OH * OW, R * S, 1);
  _im2col_tr<<<gridDim, blockDim>>>(input.cuda_buf, output.cuda_buf, OH, OW, R, S, C);
  im2col_t += (get_time() - start);
}

__global__ void _reshape_filter(float *input, float *output, size_t K, size_t C, size_t sz) {
  int rs = blockDim.x * blockIdx.x + threadIdx.x;
  int base = rs * K * C;
  for (size_t k = 0; k < K; k++) {
    for (size_t c = 0; c < C; c++) {
      int i1 = base + c * K + k;
      int i2 = base + k * C + c;
      if (i1 >= sz || i2 >= sz) printf("Illegal Access, %d %d %d\n", rs, k, c);
      output[i1] = input[i2];
    }
  }
}

void reshape_filter(Tensor *input) {
  double start = get_time();
  size_t R = input->shape[0], S = input->shape[1], K = input->shape[2], C = input->shape[3];
  input->set_buf_from_cuda();
  cudaDeviceSynchronize();
  float *tmp = (float*) malloc(K*C*sizeof(float));
  for (size_t r = 0; r < R; r++) {
    for (size_t s = 0; s < S; s++) {
      for (size_t k = 0; k < K; k++) {
        for (size_t c = 0; c < C; c++) {
          tmp[K * c + k] = input->buf[(r * S + s) * K * C + (k * C + c)];
        }
      }
      memcpy(&input->buf[(r * S + s) * K * C], tmp, K * C * sizeof(float));
    }
  }

  input->set_cuda_buf();
  reshape_t += (get_time() - start);
}


// Transposed convolution (2-dimension, stride = 2, pad = 1)
void conv2d_transposed(Tensor input, Tensor filter, Tensor bias, Tensor &output, Tensor &reshaped_input) {
  double start = get_time();
  // input shape = (in_height, in_width, in_channels)
  // filter shape = (filter_height, filter_width, output_channels, in_channels)
  // bias shape = (output_channels)
  // output shape = (in_height * stride, in_width * stride, output_channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[2];
  // assume stride 2, pad 1
  const size_t stride = 2, pad = 1;
  size_t OH = H * stride, OW = W * stride;

  output.alloc_once({OH, OW, K});
  im2col_tr(input, R, S, reshaped_input);
  shift(output, bias);
  matmul(reshaped_input, filter, output, OH * OW, K, R * S * C);
  conv2d_tr_t += (get_time() - start);
}

__global__ void _leaky_relu(float* input, float* output, float alpha) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  output[i] = (input[i] >= 0 ?  1 : alpha) * input[i];
}

// Leaky ReLU
void leaky_relu(Tensor input, Tensor &output, float alpha) {
  double start = get_time();
  // input shape = (height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  _leaky_relu<<<H*W, C>>>(input.cuda_buf, output.cuda_buf, alpha);
  leaky_relu_t += (get_time() - start);
}

__global__ void _relu(float* input, float* output) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  output[i] = input[i] >= 0 ? input[i] : 0;
}

// ReLU
void relu(Tensor input, Tensor &output) {
  double start = get_time();
  // input shape = (height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  _relu<<<H*W, C>>>(input.cuda_buf, output.cuda_buf);
  relu_t += (get_time() - start);
}

__global__ void _batchnorm(float* input, float* scale, float* offset, float* output, size_t H, size_t W, size_t C) {
  int c = blockDim.x * blockIdx.x + threadIdx.x;
  float mean = 0, sqsum = 0;
  int count = 0;
  for (size_t h = 0; h < H; ++h) {
    for (size_t w = 0; w < W; ++w) {
      float ii = input[h * W * C + w * C + c];
      count++;
      float tmp = ii - mean;
      mean += tmp/count;
      sqsum += tmp * (ii - mean);
    }
  }

  float variance = sqsum / count;
  const float epsilon = 1e-5;
  for (size_t h = 0; h < H; ++h) {
    for (size_t w = 0; w < W; ++w) {
      size_t idx = h * W * C + w * C + c;
      output[idx] = offset[c] + (input[idx] - mean) * scale[c] / sqrtf(variance + epsilon);
    }
  }

}

// Batch normalization (channel-wise)
void batchnorm(Tensor input, Tensor scale, Tensor offset, Tensor &output) {
  double start = get_time();
  // input shape = (height, width, channels)
  // scale shape = (channels)
  // offset shape = (channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  _batchnorm<<<C, 1>>>(input.cuda_buf, scale.cuda_buf, offset.cuda_buf, output.cuda_buf, H, W, C);
  batchnorm_t += (get_time() - start);
}

__global__ void _concat(float* input0, float* input1, float* output, size_t C0, size_t C1) {
  int hw = blockDim.x * blockIdx.x + threadIdx.x;
  for (size_t c = 0; c < C0; c++)
    output[hw * (C0 + C1) + c] = input0[hw * C0 + c];
  for (size_t c = 0; c < C1; ++c)
    output[hw * (C0 + C1) + (C0 + c)] = input1[hw* C1 + c];
}

// Concatenation (along channel dimension)
void concat(Tensor input0, Tensor input1, Tensor &output) {
  double start = get_time();
  // input0 shape = (height, width, channels0)
  // input1 shape = (height, width, channels1)
  // output shape = (height, width, channels0 + channels1)
  size_t H = input0.shape[0], W = input0.shape[1], C0 = input0.shape[2];
  size_t C1 = input1.shape[2];
  output.alloc_once({H, W, C0 + C1});
  _concat<<<H, W>>>(input0.cuda_buf, input1.cuda_buf, output.cuda_buf, C0, C1);
  concat_t += (get_time() - start);
}

__global__ void _elem_tanh(float* input, float* output) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  output[i] = tanhf(input[i]);
}

// Elementwise tanh
void elem_tanh(Tensor input, Tensor &output) {
  double start = get_time();
  // input shape = (height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  _elem_tanh<<<H * W, C>>>(input.cuda_buf, output.cuda_buf);
  tanh_t += (get_time() - start);
}
