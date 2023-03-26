#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>

typedef unsigned int uint;

#define LOG(msg) std::cout << msg << '\n';
#define VAR(name, value) LOG(std::setw(30) << name << ": " << value)
#define ERR(msg) std::cerr << msg << '\n';

#define CPU __host__
#define GPU __device__
#define CUDA __global__
#define APUS CPU GPU

#define KERNEL(fnc, numBlocks, numThreads) fnc<<<numBlocks,numThreads>>>

#define FAIL_NAN(var) if (isnan(var) && var != 0.0F) __trap()
#define EXIT(msg, ...) printf("Forced Exit at function: %s, line: %i. Message: \n", __FUNCTION__, __LINE__); printf(msg, __VA_ARGS__); printf("\n"); exit(1)
#define CUDA_EXIT(msg, ...) printf(msg, __VA_ARGS__); __trap()

constexpr float pi = 3.1415927F;
constexpr float f_infinity = 3.40282e+038F;

GPU void throw_error(const char* msg) {
    printf(msg);
    __trap();
}

void __trap();

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
T check(T err, const char* const func, const char* const file,
    const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
    return err;
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
cudaError_t checkLast(const char* const file, const int line)
{
    cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
    return err;
}

#define ON_ERROR_GOTO(val) cudaStatus = CHECK_CUDA_ERROR(val); if (cudaStatus != cudaSuccess) goto ERROR
#define ON_ERROR_RETURN(val) cudaStatus = CHECK_CUDA_ERROR(val); if (cudaStatus != cudaSuccess) return 1
#define RETURN_STATUS(val) cudaStatus = CHECK_CUDA_ERROR(val); if (cudaStatus != cudaSuccess) return cudaStatus