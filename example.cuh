//#include <array>
//#include <iostream>
//#include "tools.cuh"
//#include "vec.cuh"
//
//template <typename T, size_t size>
//std::ostream& operator << (std::ostream& out, const std::array<T, size> vector) {
//    out << '{';
//    const size_t max = size - 1;
//    for (int i = 0; i < max; ++i) {
//        out << vector[i] << ",";
//    }
//    out << vector[max];
//    out << '}';
//    return out;
//}
//
//class OutVec {
//public:
//    OutVec(int* data, const unsigned int size) : data(data), devPtr(0), size(size) {
//
//    }
//
//    cudaError_t allocate() {
//        return allocateGpuBuffer(&devPtr, size);
//    }
//
//    cudaError_t fromGpu() const {
//        return copyDataFromGpuBuffer(data, devPtr, size);
//    }
//
//    cudaError_t free() const {
//        return cudaFree(devPtr);
//    }
//
//    ~OutVec() {
//        free();
//    }
//
//    int* data;
//    int* devPtr;
//    const unsigned int size;
//};
//
//class InVec {
//public:
//    InVec(const int* data, const unsigned int size) : data(data), devPtr(0), size(size) {
//
//    }
//
//    cudaError_t allocate() {
//        return allocateGpuBuffer(&devPtr, size);
//    }
//
//    cudaError_t toGpu() const {
//        return copyDataToGpuBuffer(devPtr, data, size);
//    }
//
//    cudaError_t free() const {
//        return cudaFree(devPtr);
//    }
//
//    ~InVec() {
//        free();
//    }
//
//    const int* data;
//    int* devPtr;
//    const unsigned int size;
//};
//
//int test_cuda()
//{
//    const int arraySize = 5;
//    std::array<int, arraySize> a = { 1, 2, 3, 4, 5 };
//    std::array<int, arraySize> b = { 10, 20, 30, 40, 50 };
//    std::array<int, arraySize> c = { 0 };
//
//    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    //int c[arraySize] = { 0 };
//
//    InVec va(a.data(), arraySize);
//    InVec vb(b.data(), arraySize);
//    OutVec vc(c.data(), arraySize);
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(vc, va, vb, arraySize);
//    ON_ERROR(addWithCuda(vc, va, vb, arraySize), return 1);
//
//    LOG(a << " + " << b << " = " << c)
//
//        // cudaDeviceReset must be called before exiting in order for profiling and
//        // tracing tools such as Nsight and Visual Profiler to show complete traces.
//        ON_ERROR(cudaDeviceReset(), return 1);
//
//    return 0;
//}
//
////__global__ void useClass(Camera* c)
////{
////    printf("%f\n", c->origin.x);
////};
//
//int main2()
//{
//    cudaError_t cudaStatus;
//
//    Camera camera(16.0 / 9.0, 2.0, 1.0, vec3(5.4, 2.0, 3.0));
//    LOG(camera.origin.x);
//
//    GpuDataClone<Camera> c_camera(&camera);
//
//    GOTO_ERROR(c_camera.allocate());
//    GOTO_ERROR(c_camera.toGpu());
//    //useClass << <1, 1 >> > (c_camera.devPtr);
//
//    KERNEL(useClass, 1, 1)(c_camera.devPtr);
//
//
//    GOTO_ERROR(cudaDeviceSynchronize());
//ERROR:
//    c_camera.free();
//    return 0;
//}
//
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(OutVec& c, InVec& a, InVec& b, unsigned int size)
//{
//    cudaError_t cudaStatus;
//
//    cudaStatus = CHECK_LAST_CUDA_ERROR();
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    GOTO_ERROR(cudaSetDevice(0));
//
//    // Allocate GPU buffers for three vectors (two input, one output)
//    GOTO_ERROR(c.allocate());
//    GOTO_ERROR(b.allocate());
//    GOTO_ERROR(a.allocate());
//
//    // Copy input vectors from host memory to GPU buffers
//    GOTO_ERROR(a.toGpu());
//    GOTO_ERROR(b.toGpu());
//
//    // Launch a kernel on the GPU with one thread for each element.
//    // Check for any errors launching the kernel
//    addKernel << <1, size >> > (c.devPtr, a.devPtr, b.devPtr);
//
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    GOTO_ERROR(cudaDeviceSynchronize());
//
//    // Copy output vector from GPU buffer to host memory.
//    GOTO_ERROR(c.fromGpu());
//ERROR:
//    c.free();
//    b.free();
//    a.free();
//
//    return cudaStatus;
//}
//
//cudaError_t addWithCuda(OutVec& c, InVec& a, InVec& b, unsigned int size);
//
//CUDA void addKernel(int* c, const int* a, const int* b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
