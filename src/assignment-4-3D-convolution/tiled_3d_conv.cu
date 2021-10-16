#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));                   \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      return -1;                                                               \
    }                                                                          \
  } while (0)

//@@ Define program-wide constants
#define KERNEL_WIDTH 3
#define KERNEL_R KERNEL_WIDTH / 2
#define TILE_SIZE KERNEL_WIDTH
#define CACHE_SIZE (KERNEL_WIDTH + (KERNEL_R)*2)

#define checkBound(x, y, z)                                                    \
  ((0 <= x) && (x < x_size) && (0 <= y) && (y < y_size) && (0 <= z) &&         \
   (z < z_size))

//@@ Define constant memory for device kernel
__constant__ float deviceKernel[KERNEL_WIDTH * KERNEL_WIDTH * KERNEL_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Tiled convolution 3D kernel code
  int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
  int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  int tid = tz * KERNEL_WIDTH * KERNEL_WIDTH + ty * KERNEL_WIDTH + tx;
  // Allocate shared memory for a tile cache
  __shared__ float cache[CACHE_SIZE][CACHE_SIZE][CACHE_SIZE];
  if (tid < CACHE_SIZE * CACHE_SIZE) {
    int tile_y = (tid / CACHE_SIZE) % CACHE_SIZE;
    int tile_x = tid % CACHE_SIZE;

    int z = bz * TILE_SIZE - 1;
    int y = by * TILE_SIZE + tile_y - 1;
    int x = bx * TILE_SIZE + tile_x - 1;
    for (int i = 0; i < CACHE_SIZE; i++) {
      int z_tmp = z + i;
      cache[tile_x][tile_y][i] =
          checkBound(x, y, z_tmp)
              ? input[z_tmp * y_size * x_size + y * x_size + x]
              : 0;
    }
  }

  __syncthreads();

  int x_ = bx * TILE_SIZE + tx;
  int y_ = by * TILE_SIZE + ty;
  int z_ = bz * TILE_SIZE + tz;

  if (checkBound(x_, y_, z_)) {
    float val = 0;
    for (int x = 0; x < KERNEL_WIDTH; x += 1) {
      for (int y = 0; y < KERNEL_WIDTH; y += 1) {
        for (int z = 0; z < KERNEL_WIDTH; z += 1) {
          val += cache[tx + x][ty + y][tz + z] *
                 deviceKernel[z * KERNEL_WIDTH * KERNEL_WIDTH +
                              y * KERNEL_WIDTH + x];
        }
      }
    }
    output[z_ * y_size * x_size + y_ * x_size + x_] = val;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  int size = z_size * y_size * x_size * sizeof(float);
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  cudaMalloc((void **)&deviceOutput, size);
  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocating GPU memory
  cudaMalloc((void **)&deviceOutput, size);
  cudaMalloc((void **)&deviceInput, size);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU
  cudaMemcpy(deviceInput, hostInput + 3, size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
  dim3 dimGrid(ceil(float(x_size) / TILE_SIZE), ceil(float(y_size) / TILE_SIZE),
               ceil(float(z_size) / TILE_SIZE));
  //@@ Launch the GPU kernel
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size,
                                x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host
  cudaMemcpy(hostOutput + 3, deviceOutput, size, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
