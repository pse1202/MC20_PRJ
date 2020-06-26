#pragma once

#include <cstddef>
#include <cstdint>

void _pix2pix(uint8_t *input_buf, float *weight_buf, uint8_t *output_buf, size_t num_image);