#include "pix2pix.h"
#include "kernel.h"
#include "util.h"

#ifdef USE_MPI
#include <mpi.h>
#endif
#include <stdio.h>

static size_t IMAGESIZE = (256 * 256 * 3);
static size_t WEIGHTSIZE = 217725452 / sizeof(float);

void pix2pix(uint8_t *input_buf, float *weight_buf, uint8_t *output_buf, size_t tot_images) {
    /*
     * !!!!!!!! Caution !!!!!!!!
     * In MPI program, all buffers and num_image are only given to rank 0 process.
     * You should manually:
     *   1. allocate buffers on others
     *   2. send inputs from rank 0 to others
     *   3. gather outputs from others to rank 0
     */
    
    int num_image;
    int rank = get_rank();
  
    if (rank == 0) {
      // send part of image, all weights, part num_image, malloc output_buf
      int cumsum = (tot_images / 4) + ((tot_images % 4 > 0 ? 1 : 0));
      for (int i = 1; i < 4; i++) {
        int currnum = (tot_images / 4) + ((tot_images % 4 > i ? 1 : 0));
        MPI_Send(weight_buf, WEIGHTSIZE , MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        MPI_Send(&currnum, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(&(input_buf[cumsum * IMAGESIZE]), IMAGESIZE * currnum, MPI_UINT8_T, i, 0, MPI_COMM_WORLD);
        cumsum += currnum;
      }
      num_image = (tot_images / 4) + ((tot_images % 4 > 0 ? 1 : 0));
    } else {
      weight_buf = (float*)malloc(217725452);
      MPI_Recv(weight_buf, WEIGHTSIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
      int num_buf[1];
      MPI_Recv(num_buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
      num_image = num_buf[0];
      input_buf = (uint8_t*)malloc(num_image * IMAGESIZE);
      MPI_Recv(input_buf, IMAGESIZE * num_image, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, NULL);
      output_buf = (uint8_t*)malloc(num_image * IMAGESIZE);
    }
  
    _pix2pix(input_buf, weight_buf, output_buf, num_image);
  
    if (rank == 0) {
      int curr = num_image * IMAGESIZE;
      for (int i = 1; i < 4; i++) {
        int currnum = ((tot_images / 4) + ((tot_images % 4 > i ? 1 : 0))) * IMAGESIZE;
        MPI_Recv(&(output_buf[curr]), currnum, MPI_UINT8_T, i, 0, MPI_COMM_WORLD, NULL);
        curr += currnum;
      }
    } else {
      MPI_Send(output_buf, num_image * IMAGESIZE, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD);
    }
  }

  void pix2pix_init() {
    /*
     * You can do input-independent and input-size-independent jobs here.
     * e.g., Getting OpenCL platform, Compiling OpenCL kernel, ...
     * Execution time of this function is not measured, so do as much as possible!
     */ 
  }