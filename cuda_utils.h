#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdio>
#include <cuda_runtime.h>

inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
    }
}

#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

template<typename T>
void deviceAlloc(T** ptr, size_t count) {
    CUDA_CHECK(cudaMalloc(ptr, count * sizeof(T)));
}

template<typename T>
void copyToDevice(T* dst, T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void copyToHost(T* dst, T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void deviceFree(T* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

#endif