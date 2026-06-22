#include <cstdio>
#include <fstream>
#include <chrono>
#include "cuda_utils.h"

__device__ __host__ int mandelbrotIterations(double cx, double cy, int maxIter) {
    double zx = 0.0, zy = 0.0;
    int iter = 0;
    
    while (zx*zx + zy*zy <= 4.0 && iter < maxIter) {
        double newZx = zx * zx - zy * zy + cx;
        double newZy = 2 * zx * zy + cy;
        zx = newZx;
        zy = newZy;
        iter++;
    }

    return iter;
}

__device__ __host__ void iterToColor(int iter, int maxIter, unsigned char& r, unsigned char& g, unsigned char& b) {
    if (iter == maxIter) {
        r = g = b = 0;
        return;
    }

    double t = double (iter) / maxIter;

    r = (unsigned char) (9 * (1 - t) * t * t * t * 255);
    g = (unsigned char) (15 * (1 - t) * (1 - t) * t * t * 255);
    b = (unsigned char) (8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
}

__global__ void mandelbrotKernel(
    unsigned char* image,
    int width, int height, int maxIter,
    double xmin, double xmax, double ymin, double ymax
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px < width && py < height) {
        double cr = xmin + (px / (double) width) * (xmax - xmin);
        double ci = ymin + (py / (double) height) * (ymax - ymin);

        int iter = mandelbrotIterations(cr, ci, maxIter);

        int idx = (py * width + px) * 3;
        iterToColor(iter, maxIter, image[idx + 0], image[idx + 1], image[idx + 2]);
    }
}

void mandelbrotCPU(unsigned char* image, int width, int height, int maxIter,
                    double xmin, double xmax, double ymin, double ymax) {
    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {
            double cr = xmin + (px / (double)width)  * (xmax - xmin);
            double ci = ymin + (py / (double)height) * (ymax - ymin);

            int iter = mandelbrotIterations(cr, ci, maxIter);

            int idx = (py * width + px) * 3;
            iterToColor(iter, maxIter, image[idx+0], image[idx+1], image[idx+2]);
        }
    }
}

int main() {
    int width = 800, height = 600;
    int maxIter = 256;
    double xmin = -2.5, xmax = -1;
    double ymin = -1.0, ymax = 1.0;

    size_t bytes = (size_t) width * height * 3;

    unsigned char* h_image_cpu = new unsigned char[bytes];
    unsigned char* h_image_gpu = new unsigned char[bytes];

    auto cpuStart = std::chrono::high_resolution_clock::now();
    mandelbrotCPU(h_image_cpu, width, height, maxIter, xmin, xmax, ymin, ymax);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    printf("CPU time: %f ms\n", cpuMs);

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

    mandelbrotKernel<<<blocks, threadsPerBlock>>>(
        d_image, width, height, maxIter, xmin, xmax, ymin, ymax
    );
    cudaEventRecord(gpuEnd);
    cudaEventSynchronize(gpuEnd); 
    CUDA_CHECK(cudaGetLastError());

    float gpuMs = 0;
    cudaEventElapsedTime(&gpuMs, gpuStart, gpuEnd);
    printf("GPU kernel time: %f ms\n", gpuMs);
    printf("Speedup: %fx\n", cpuMs / gpuMs);

    copyToHost(h_image_gpu, d_image, bytes);

    std::ofstream outCpu("output/mandelbrot_cpu.ppm", std::ios::binary);
    outCpu << "P6\n" << width << " " << height << "\n255\n";
    outCpu.write((char*) h_image_cpu, bytes);

    std::ofstream outGpu("output/mandelbrot_gpu.ppm", std::ios::binary);
    outGpu << "P6\n" << width << " " << height << "\n255\n";
    outGpu.write((char*)h_image_gpu, bytes);

    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuEnd);
    deviceFree(d_image);
    delete[] h_image_cpu;
    delete[] h_image_gpu;

    return 0;
}