// ssim_gpu.cpp
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void ssim_kernel(const float* img1, const float* img2, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        // Calculate mean and variance of image 1
        float mean1 = 0.0f, var1 = 0.0f;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int nx = x + i, ny = y + j;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float val = img1[ny * width + nx];
                    mean1 += val;
                    var1 += val * val;
                }
            }
        }
        mean1 /= 9.0f;
        var1 = var1 / 9.0f - mean1 * mean1;

        // Calculate mean and variance of image 2
        float mean2 = 0.0f, var2 = 0.0f;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int nx = x + i, ny = y + j;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float val = img2[ny * width + nx];
                    mean2 += val;
                    var2 += val * val;
                }
            }
        }
        mean2 /= 9.0f;
        var2 = var2 / 9.0f - mean2 * mean2;

        // SSIM
        float c1 = 0.01f, c2 = 0.03f;
        float ssim = (2 * mean1 * mean2 + c1) * (2 * sqrt(var1 * var2) + c2) /
                    ((mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2));
        out[idx] = ssim;
    }
}

extern "C" __global__ void ssim_gpu(const float* img1, const float* img2, float* out, int width, int height) {
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    ssim_kernel<<<grid_size, block_size>>>(img1, img2, out, width, height);
}
