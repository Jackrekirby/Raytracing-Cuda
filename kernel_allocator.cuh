#include "tools.cuh"

class KernelAllocator {
public:
    KernelAllocator(int width, int height) {
        int size = width * height;
        this->width = width;
        this->height = height;
        num_threads = std::min(size, max_threads);
        num_blocks = size / max_threads + 1;
    }

    GPU int2 get_coords(int iThread, int iBlock) const {
        int2 coords = { 0, 0 };
        int i = iThread + iBlock * max_threads;
        coords.y = i / width;
        coords.x = i - coords.y * width;
        return coords;
    }

    static const int max_threads = 1024;
    int num_threads;
    int num_blocks;
    int width;
    int height;
};