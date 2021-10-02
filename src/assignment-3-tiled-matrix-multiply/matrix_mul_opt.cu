
#include <wb.h>
#define BLK_DIM 32
#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  int Width = numAColumns;
  __shared__ float aTile[BLK_DIM][BLK_DIM],
      bTile[BLK_DIM][BLK_DIM]; // block dim set to 32.
  float entry = 0;
  int by = blockIdx.y;
  int bx = blockIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int y = BLK_DIM * by + ty;
  int x = BLK_DIM * bx + tx;
  for (int i = 0; i < (Width - 1) / BLK_DIM + 1; ++i) {
    if (y < numARows && (i * BLK_DIM + tx < numAColumns)) {
      float *Asub = A + numAColumns * by * BLK_DIM + i * BLK_DIM;
      aTile[ty][tx] = Asub[ty * numAColumns + tx];
    } else
      aTile[ty][tx] = 0;
    if (i * BLK_DIM + ty < numBRows && x < numBColumns) {
      float *Bsub = B + numBColumns * i * BLK_DIM + bx * BLK_DIM;
      bTile[ty][tx] = Bsub[ty * numBColumns + tx];
    } else
      bTile[ty][tx] = 0;
    __syncthreads();
    for (int k = 0; k < BLK_DIM; ++k) {
      entry += aTile[ty][k] * bTile[k][tx];
    }
    __syncthreads();
  }
  if (y < numCRows && x < numCColumns)
    C[y * numCColumns + x] = entry;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  wbCheck(cudaMalloc(&deviceA, numARows * numAColumns * sizeof(float)));
  wbCheck(cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(float)));
  wbCheck(cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlk(BLK_DIM, BLK_DIM);
  dim3 dimGrid(numCColumns % dimBlk.x ? numCColumns / dimBlk.x + 1
                                      : numCColumns / dimBlk.x,
               numCRows % dimBlk.y ? numCRows / dimBlk.y + 1
                                   : numCRows / dimBlk.y);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlk>>>(deviceA, deviceB, deviceC, numARows,
                                            numAColumns, numBRows, numBColumns,
                                            numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
