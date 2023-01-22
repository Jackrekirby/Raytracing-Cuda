#pragma once
#include <vector>
#include <string>
#include <fstream>

class Image {
public:
    Image(int width, int height) 
        : width(width), height(height),
        aspect_ratio(static_cast<float>(width) / static_cast<float>(height)) {
        create_pixels();
    }

    Image(int width, float aspect_ratio = 16.0 / 9.0)
        : width(width), height(static_cast<int>(width / aspect_ratio)),
        aspect_ratio(aspect_ratio) {
        create_pixels();
    }

    void create_pixels() {
        const int size = width * height;
        pixels.reserve(size);
        for (int i = 0; i < size; ++i) {
            pixels.push_back(Char3(255, 255, 255));
        }
    }

    void save_as_ppm(const std::string& name) {
        std::ofstream file;
        file.open(name + ".ppm");
        file << "P3\n" << width << ' ' << height << "\n255\n";
        const int size = static_cast<int>(pixels.size());
        for (int i = 0; i < size; i += 1) {
            const auto &pixel = pixels[i];
            file << +pixel.x << ' ' << +pixel.y << ' ' << +pixel.z << '\n';
        }
        file.close();
    }

    void save_as_bin(const std::string& name) {
        std::ofstream file(name + ".bin", std::ios::binary);
        file.write((char*)pixels.data(), sizeof(Char3) * pixels.size());
        file.close();
    }

    int width;
    int height;
    float aspect_ratio;
    std::vector<Char3> pixels;
};
