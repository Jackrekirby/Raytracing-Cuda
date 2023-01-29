#pragma once
#include <array>
#include "materials.cuh"
#include "object.cuh"

template <int n>
std::array<Metal, n> build_metals(int& seed) {
    std::array<Metal, n> metals{};

    for (int i = 0; i < n; i++) {
        auto& metal = metals[i];
        metal.color = Float3::random(seed, 0.5, 1);
        metal.roughness = random_float(seed, 0, 1);
    }

    return metals;
}

template <int n>
std::array<Lambert, n> build_lamberts(int& seed) {
    std::array<Lambert, n> lamberts{};
    for (int i = 0; i < n; i++) {
        auto& lambert = lamberts[i];
        lambert.color = Float3::random(seed, 0.5, 1);
    }
    return lamberts;
}

template <int n>
std::array<Dielectric, n> build_dielectrics(int& seed) {
    std::array<Dielectric, n> dielectrics{};

    for (int i = 0; i < n; i++) {
        auto& dielectric = dielectrics[i];
        dielectric.color = Float3::random(seed, 0.5, 1);
        dielectric.refractive_index = random_float(seed, 0.5, 4);
    }
    return dielectrics;
}

template <int n_lamberts, int n_metals, int n_dielectrics>
void build_scene() {
    // user variables
    int seed = 1738463;
    const int n_objects = 1000;
    const int max_propensity = 100;
    const int n_materials = 3;
    std::array<int, n_materials> material_propensities = { 50, 25, 25 };

    std::array<int, n_materials> material_counts{};

    // check material propensities sum to max_propensity
    int total_propensity = 0;
    for (auto propensity : material_propensities) {
        total_propensity += propensity;
    }
    if (total_propensity != max_propensity) {
        printf("material propensities don't sum to max propensity");
        return;
    }

    // convert propensities into actual number of material variants
    int current_total_objects = 0;
    for (int i = 1; i < n_materials; ++i) {
        int n = static_cast<int>(static_cast<float>(n_objects) * static_cast<float>(material_propensities[i]) / static_cast<float>(max_propensity));
        material_counts[i] = n;
        current_total_objects += n;
    }
    material_counts[0] = n_objects - current_total_objects;

    // create object ptrs
    std::array<ObjectPtr, n_objects> objectPtrs{};

    int objects_remaining = n_objects;
    for (int i = 0; i < n_objects; ++i) {
        float r = random_float(seed);
        float r_check = 0;
        auto& objectPtr = objectPtrs[i];

        objectPtr.shape = Shape::sphere;
        objectPtr.shapeIndex = i;

        for (int j = 0; j < n_materials; ++j) {
            r_check += static_cast<float>(material_counts[j]) / static_cast<float>(objects_remaining);

            if (r < r_check) {
                material_counts[i] -= 1;
                objects_remaining -= 1;
                objectPtr.material = static_cast<Material>(j);
                objectPtr.materialIndex = material_counts[i];
            }
        }
    }

    auto lamberts = build_lamberts<n_lamberts>(seed);
    auto metals = build_metals<n_metals>(seed);
    auto dielectrics = build_dielectrics<n_dielectrics>(seed);
}