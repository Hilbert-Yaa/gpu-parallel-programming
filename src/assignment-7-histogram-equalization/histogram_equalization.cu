// Histogram Equalization

#include <wb.h>

#define BLOCK_SIZE 1024
#define HISTOGRAM_SIZE 256

__global__ void RGB2GS(unsigned char *output, float *input, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = tid / 3;
  if (idx < size) {
    unsigned char r = (unsigned char)(255 * input[idx * 3]);
    unsigned char g = (unsigned char)(255 * input[idx * 3 + 1]);
    unsigned char b = (unsigned char)(255 * input[idx * 3 + 2]);
    output[idx] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
  }
  __syncthreads();
}

__global__ void GS2RGB(float *output, unsigned char *input, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = tid / 3;
  if (idx < size) {
    float tmp = input[idx] / 255.0;
    output[3 * idx] = tmp;
    output[3 * idx + 1] = tmp;
    output[3 * idx + 2] = tmp;
  }
  __syncthreads();
}

__global__ void calcHistogram(float *histogram, unsigned char *input, int size,
                              bool priv) {
  if (priv) {
    // privatization opt, faster atomic Op with smem
    __shared__ float cache[HISTOGRAM_SIZE];
    if (threadIdx.x < HISTOGRAM_SIZE)
      cache[threadIdx.x] = 0;
    __syncthreads();
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < size) {
      atomicAdd(&(cache[input[tid]]), 1);
      tid += stride;
    }
    __syncthreads();
    if (threadIdx.x < HISTOGRAM_SIZE)
      atomicAdd(&(histogram[threadIdx.x]), cache[threadIdx.x]);
  } else {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < size) {
      atomicAdd(&(histogram[input[tid]]), 1);
      tid += stride;
    }
  }
}

__global__ void calcCDF(float *cdf, float *histogram, int size) {
    // Kogge-Stone Scan
    __shared__ float scan[HISTOGRAM_SIZE];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
      scan[threadIdx.x] = histogram[tid];
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
      __syncthreads();
      if (threadIdx.x >= stride) {
        scan[threadIdx.x] += scan[threadIdx.x - stride];
      }
    }
    __syncthreads();
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
      cdf[i] = scan[i] / size;
    }
}

__global__ void equalHistogram(float *outputImageData, float *inputImageData,
                                  float *cdf, int size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  float minimum = cdf[0];
  if (tid < size) {
    unsigned char tmp = (unsigned char)(255 * inputImageData[tid]);
    tmp = 255 * (cdf[tmp] - minimum) / (1 - minimum);
    tmp = min(255, tmp);
    outputImageData[tid] = tmp / 255.0;
  }
  __syncthreads();
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  int imageSize;
  int gramDataSize;
  const char *inputImageFile;
  // declare host-side var
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  // declare device-side var
  float *deviceInputImageData;
  unsigned char *deviceGSImageData;
  float *deviceHistogram;
  float *cdf;
  float *deviceOutputImageData;
  // import args, input files and get params
  args = wbArg_read(argc, argv);
  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage = wbImport(inputImageFile);
  hostInputImageData = wbImage_getData(inputImage);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage);
  imageSize = imageWidth * imageHeight;
  size_t imageDataSize = imageSize * imageChannels * sizeof(float);
  gramDataSize = HISTOGRAM_SIZE * sizeof(float);
  // device memory allocation & copy
  cudaMalloc((void **)&deviceInputImageData, imageDataSize);
  cudaMalloc((void **)&deviceGSImageData, imageSize);
  cudaMalloc((void **)&deviceHistogram, gramDataSize);
  cudaMalloc((void **)&cdf, gramDataSize);
  cudaMalloc((void **)&deviceOutputImageData, imageDataSize);
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageDataSize,
             cudaMemcpyHostToDevice);
  int gridDimRGB = ceil(float(imageSize * imageChannels) / BLOCK_SIZE);
  int gridDimGS = ceil(float(imageSize) / BLOCK_SIZE);
  // launch CUDA kernels
  RGB2GS<<<gridDimRGB, BLOCK_SIZE>>>(deviceGSImageData, deviceInputImageData, 
                                     imageSize);
  // GS2RGB<<<gridDim, BLOCK_SIZE>>> (deviceGSImageData, deviceOutputImageData,
  // imageSize); // debug-only
  calcHistogram<<<gridDimGS, BLOCK_SIZE>>>(deviceHistogram, deviceGSImageData, 
                                           imageSize, true);
  calcCDF<<<1, HISTOGRAM_SIZE>>>(
      cdf, deviceHistogram, imageSize);
  equalHistogram<<<gridDimRGB, BLOCK_SIZE>>>(deviceOutputImageData,
                                                deviceInputImageData, cdf,
                                                imageSize * imageChannels);
  // transfer results deivce to host
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageDataSize,
             cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  // free allocated memory
  cudaFree(deviceInputImageData);
  cudaFree(deviceHistogram);
  cudaFree(cdf);
  cudaFree(deviceOutputImageData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
