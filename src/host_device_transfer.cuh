#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <array>
#include <vector>

template <typename T>
cudaError_t allocateGpuBuffer(T** devPtr, unsigned int size = 1) {
    return cudaMalloc((void**)devPtr, size * sizeof(T));
}

template <typename T>
cudaError_t copyDataToGpuBuffer(T* dst, const T* src, unsigned int size = 1) {
    return cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
cudaError_t copyDataFromGpuBuffer(T* dst, const T* src, unsigned int size = 1) {
    return cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost);
}


template <typename T>
class GpuDataClone {
public:

    GpuDataClone(T* hostPtr, int size = 1) : hostPtr(hostPtr), devPtr(0), size(size) {

    }

    template <int N>
    GpuDataClone(std::array<T, N>& list) : hostPtr(list.data()), devPtr(0), size(N) {

    }

    GpuDataClone(std::vector<T>& list) : hostPtr(list.data()), devPtr(0), size(static_cast<int>(list.size())) {

    }

    cudaError_t allocate() {
        return allocateGpuBuffer(&devPtr, size);
    }

    cudaError_t toGpu() const {
        return copyDataToGpuBuffer(devPtr, hostPtr, size);
    }

    cudaError_t fromGpu() const {
        return copyDataFromGpuBuffer(hostPtr, devPtr, size);
    }

    cudaError_t free() const {
        return cudaFree(devPtr);
    }

    ~GpuDataClone() {
        free();
    }

    T* hostPtr;
    T* devPtr;
    const unsigned int size;
};