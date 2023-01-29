#include "tools.cuh"
#include "host_device_transfer.cuh"

template <typename T, int N>
class Array : public GpuDataClone<Array<T, N>> {
public:
    Array(std::initializer_list<T> data) : GpuDataClone<Array<T, N>>(this), size(N) {
        /*      if (data.size() > N) {
                  EXIT("initialiser size (%i) > array size (%i)", data.size(), N);
              }*/
        for (int i = 0; i < data.size(); ++i) {
            this->data[i] = *(data.begin() + i);
        }
    }

    Array() : size(N) {

    }

    APUS void fill(const T& val) {
        for (int i = 0; i < N; ++i) {
            data[i] = val;
        }
    }

    APUS T operator [](int i) const {
        return data[i];
    }

    APUS T& operator [](int i) {
        return data[i];
    }

    T data[N];
    int size = N;
};

template <typename T>
class List {
public:
    List(std::initializer_list<T> data) : size(data.size()) {
        for (int i = 0; i < data.size(); ++i) {
            this->data[i] = *(data.begin() + i);
        }
    }

    List(int size) : size(size) {

    }

    APUS void fill(const T& val) {
        for (int i = 0; i < N; ++i) {
            data[i] = val;
        }
    }

    APUS T operator [](int i) const {
        return data[i];
    }

    APUS T& operator [](int i) {
        return data[i];
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

    T* data;
    int size;
};

CUDA void testkernal(Array<int, 5>* pixels_ptr) {
    auto& pixels = *pixels_ptr;
    for (int i = 0; i < pixels.size; ++i) {
        printf("%i\n", pixels[i]);
    }
}

CPU void test_array() {
    List<int> a = { 1, 2, 3, 4 };
    //Array<int, 5> a = { 1, 2, 3, 4, 5 };
    cudaError_t cudaStatus;

    ON_ERROR_GOTO(a.allocate());
    ON_ERROR_GOTO(a.toGpu());

    for (uint i = 0; i < a.size; ++i) {
        std::cout << a[i] << '\n';
    }

    //testkernal(&a);
    KERNEL(testkernal, 1, 1)(a.devPtr);

    ON_ERROR_GOTO(cudaDeviceSynchronize());

ERROR:
    a.free();
}