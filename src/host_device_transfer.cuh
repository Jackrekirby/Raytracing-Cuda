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

// Alien = a entity which lives on both cpu and gpu

class Alien {
public:
    virtual cudaError_t allocate() = 0;

    virtual cudaError_t toGpu() const = 0;

    virtual cudaError_t fromGpu() const = 0;

    virtual cudaError_t free() const = 0;

};

template <typename T>
class GpuDataClone : public Alien {
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

enum class AlienType {
    IN, OUT, INOUT
};

struct AlienWrapper {
    Alien* alien;
    AlienType type;
};

class AlienManager {
public:
    template <typename T>
    GpuDataClone<T>& add(T& data, AlienType type = AlienType::IN) {
        GpuDataClone<T>* clone = new GpuDataClone<T>(&data);
        Alien* alien = clone;
        aliens.push_back({ alien, type });
        return *clone;
    }

    template <typename T, int N>
    GpuDataClone<T>& add(std::array<T, N>& list, AlienType type = AlienType::IN) {
        GpuDataClone<T>* clone = new GpuDataClone<T>(list);
        Alien* alien = clone;
        aliens.push_back({ alien, type});
        return *clone;
    }

    template <typename T>
    GpuDataClone<T>& add(std::vector<T>& list, AlienType type = AlienType::IN) {
        GpuDataClone<T>* clone = new GpuDataClone<T>(list);
        Alien* alien = clone;
        aliens.push_back({ alien, type });
        return *clone;
    }

    cudaError_t allocate() {
        cudaError_t cudaStatus;
        for (const auto& aw : aliens) {
            RETURN_STATUS(aw.alien->allocate());
        }
        return cudaError::cudaSuccess;
    }

    cudaError_t toGpu() {
        cudaError_t cudaStatus;
        for (const auto &aw : aliens) {
            if (aw.type == AlienType::IN || aw.type == AlienType::INOUT) {
                RETURN_STATUS(aw.alien->toGpu());
            }
        }
        return cudaError::cudaSuccess;
    }

    cudaError_t fromGpu() {
        cudaError_t cudaStatus;
        for (const auto& aw : aliens) {
            if (aw.type == AlienType::OUT || aw.type == AlienType::INOUT) {
                RETURN_STATUS(aw.alien->fromGpu());
            }
        }
        return cudaError::cudaSuccess;
    }

    // doesnt catch cudaError
    void free() const {
        for (const auto& aw : aliens) {
            aw.alien->free(); // free gpu memory
            //delete aw.alien; // free cpu memory
        }
    }

    ~AlienManager() {
        free();
    }

    std::vector<AlienWrapper> aliens;
};

