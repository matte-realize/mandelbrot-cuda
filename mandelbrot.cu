#include <cstdio>
#include <fstream>
#include "cuda_utils.h"

__device__ __host__ double mandelbrotSmooth(double cx, double cy, int maxIter) {
    double zx = 0.0, zy = 0.0;
    int iter = 0;

    while (zx * zx + zy * zy <= 4.0 && iter < maxIter) {
        double newZx = zx * zx - zy * zy + cx;
        double newZy = 2 * zx * zy + cy;
        zx = newZx;
        zy = newZy;
        iter++;
    }

    if (iter == maxIter) {
        return (double) maxIter;
    }

    double zn = sqrt(zx * zx + zy * zy);
    double nu = log(log(zn) / log(2.0)) / log(2.0);
    return (double) iter + 1.0 - nu;
}

__device__ __host__ void iterToColorSmooth(double smoothIter, int maxIter, unsigned char& r, unsigned char& g, unsigned char& b) {
    if (smoothIter >= maxIter) {
        r = g = b = 0;
        return;
    }

    double t = smoothIter / maxIter;

    r = (unsigned char) (9 * (1 - t) * t * t * t * 255);
    g = (unsigned char) (15 * (1 - t) * (1 - t) * t * t * 255);
    b = (unsigned char) (8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
}

__global__ void mandelbrotKernelSupersampled(
    unsigned char* image,
    int width, int height, int maxIter,
    double xmin, double xmax, double ymin, double ymax,
    int samplesPerAxis
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px < width && py < height) {
        double rSum = 0.0, gSum = 0.0, bSum = 0.0;
        int totalSamples = samplesPerAxis * samplesPerAxis;

        for (int sy = 0; sy < samplesPerAxis; sy++) {
            for (int sx = 0; sx < samplesPerAxis; sx++) {
                double offsetX = (sx + 0.5) / samplesPerAxis;
                double offsetY = (sy + 0.5) / samplesPerAxis;

                double cr = xmin + ((px + offsetX) / (double) width) * (xmax - xmin);
                double ci = ymin + ((py + offsetY) / (double) height) * (ymax - ymin);

                double iter = mandelbrotSmooth(cr, ci, maxIter);

                unsigned char r, g, b;
                iterToColorSmooth(iter, maxIter, r, g, b);

                rSum += r;
                gSum += g;
                bSum += b;
            }
        }

        int idx = (py * width + px) * 3;
        image[idx + 0] = (unsigned char) (rSum / totalSamples);
        image[idx + 1] = (unsigned char) (gSum / totalSamples);
        image[idx + 2] = (unsigned char) (bSum / totalSamples);
    }
}

int main() {
    int width = 800, height = 600;
    int maxIter = 20;
    double xmin = -2.5, xmax = -1;
    double ymin = -1.0, ymax = 1.0;
    int samplesPerAxis = 20;

    size_t bytes = (size_t) width * height * 3;
    unsigned char* h_image = new unsigned char[bytes];

    unsigned char* d_image;
    deviceAlloc(&d_image, bytes);

    dim3 threadsPerBlock(16, 16);
    dim3 blocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    cudaEvent_t gpuStart, gpuEnd;
    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuEnd);

    cudaEventRecord(gpuStart);
    mandelbrotKernelSupersampled<<<blocks, threadsPerBlock>>>(
        d_image, width, height, maxIter, xmin, xmax, ymin, ymax, samplesPerAxis
    );
    cudaEventRecord(gpuEnd);
    cudaEventSynchronize(gpuEnd);
    CUDA_CHECK(cudaGetLastError());

    float gpuMs = 0;
    cudaEventElapsedTime(&gpuMs, gpuStart, gpuEnd);
    printf("Render time: %f ms\n", gpuMs);

    copyToHost(h_image, d_image, bytes);

    std::ofstream out("output/mandelbrot.ppm", std::ios::binary);
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write((char*) h_image, bytes);

    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuEnd);
    deviceFree(d_image);
    delete[] h_image;

    return 0;
}