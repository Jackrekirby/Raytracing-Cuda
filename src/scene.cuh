#pragma once
#include <array>
#include <vector>
#include "materials.cuh"
#include "object.cuh"

constexpr uint n_materials = 3, n_shapes = 1;
class ObjectManager {
public:
    ObjectManager(uint n_objects, uint n_spheres, uint n_lamberts, uint n_metals, uint n_dielectrics):
        count_objects(0), total_objects(n_objects) {
        if (n_objects != n_spheres) {
            EXIT("n_objects != n_spheres");
        }
        if (n_objects != (n_lamberts + n_metals + n_dielectrics)) {
            EXIT("n_objects != (n_lamberts + n_metals + n_dielectrics)");
        }

        total_materials = { n_lamberts, n_metals, n_dielectrics };
        total_shapes = { n_spheres };

        count_materials = { 0, 0, 0 };
        count_shapes = { 0 };
    }

    void add_object(Shape shape, Material material) {
        uint &material_count = count_materials[static_cast<uint>(material)];
        uint& shape_count = count_shapes[static_cast<uint>(shape)];
        objectPtrs.push_back({ shape, shape_count, material, material_count });
        material_count++;
        shape_count++;
        count_objects++;
    }

    void build_objects() {
        objects = Objects(objectPtrs.data(), spheres.data(), lamberts.data(), metals.data(), dielectrics.data());
    }

    void random_fill(
        int& seed,
        const Float3 offset,
        float max_placement_radius,
        float min_radius,
        float max_radius,
        int max_attempts
    ) {
        while(count_objects < total_objects) {
            float r = random_float(seed);
            float r_check = 0;

            for (int i = 0; i < n_materials; ++i) {
                const int material_left = total_materials[i] - count_materials[i];
                const int objects_left = total_objects - count_objects;
                r_check += static_cast<float>(material_left) / static_cast<float>(objects_left);
                //printf("%i, %i, %.4f\n", material_left, objects_left, r_check);

                if (r < r_check) {
                    bool was_made = false;
                    for (int j = 0; j < max_attempts; ++j) {
                        Sphere sphere = random_sphere(seed, offset, max_placement_radius, min_radius, max_radius);
                        was_made = add_sphere(sphere);
                        if (was_made) break;
                    }

                    if (!was_made) {
                        EXIT("Failed to place sphere");
                        return; // if failed to place sphere many times then give up
                    }

                    Material material = static_cast<Material>(i);
                    add_object(Shape::sphere, material);

                    switch (material)
                    {
                    case Material::Lambert:
                        lamberts.push_back(random_lambert(seed));
                        break;
                    case Material::Metal:
                        metals.push_back(random_metal(seed));
                        break;
                    case Material::Dielectric:
                        dielectrics.push_back(random_dielectric(seed));
                        break;
                    }
                    break;
                }
            }
        }
    }

    Sphere random_sphere(
        int& seed, 
        const Float3 offset,
        float max_placement_radius,
        float min_radius,
        float max_radius
    ) {
        float radius = random_float(seed, min_radius, max_radius);
        float placement_radius = max_placement_radius * sqrt(random_float(seed));
        float placement_angle = random_float(seed) * 2 * pi;

        Float3 center = Float3(
            placement_radius * cosf(placement_angle),
            radius,
            placement_radius * sinf(placement_angle)
        ) + offset;

        return { center, radius };
    }

    Dielectric random_dielectric(int &seed) {
        const auto color = Float3::random(seed, 0.5, 1);
        const auto refractive_index = random_float(seed, 0.5, 4);
        return { color, refractive_index };
    }

    Lambert random_lambert(int& seed) {
        return { Float3::random(seed, 0.5, 1) };
    }

    Metal random_metal(int& seed) {
        const auto color = Float3::random(seed, 0.5, 1);
        const auto roughness = random_float(seed, 0, 1);
        return { color, roughness };
    }


    // fail if intersecting another sphere
    bool add_sphere(const Sphere& sphere) {
        bool can_place = true;

        for (auto &other : spheres) {
            float min_dist = sphere.radius + other.radius;
            if ((sphere.center - other.center).length_squared() < min_dist * min_dist) {
                can_place = false;
                break;
            }
        }

        if (can_place) {
            spheres.push_back(sphere);
        }
        return can_place;
    }

    int total_objects;
    int count_objects;
    std::array<uint, n_shapes> total_shapes;
    std::array<uint, n_shapes> count_shapes;
    std::array<uint, n_materials> total_materials;
    std::array<uint, n_materials> count_materials;

    std::vector<ObjectPtr> objectPtrs;

    std::vector<Sphere> spheres;
    std::vector<Lambert> lamberts;
    std::vector<Metal> metals;
    std::vector<Dielectric> dielectrics;
    Objects objects;
};


//auto lamberts = build_lamberts<n_lamberts>(seed);
//auto metals = build_metals<n_metals>(seed);
//auto dielectrics = build_dielectrics<n_dielectrics>(seed);


//// check material propensities sum to max_propensity
//int total_propensity = 0;
//for (auto propensity : material_propensities) {
//    total_propensity += propensity;
//}
//if (total_propensity != max_propensity) {
//    printf("material propensities don't sum to max propensity");
//    return;
//}
//
//// convert propensities into actual number of material variants
//int current_total_objects = 0;
//std::array<int, n_materials> material_counts{};
//for (int i = 1; i < n_materials; ++i) {
//    int n = static_cast<int>(static_cast<float>(n_objects) * static_cast<float>(material_propensities[i]) / static_cast<float>(max_propensity));
//    material_counts[i] = n;
//    current_total_objects += n;
//}
//material_counts[0] = n_objects - current_total_objects;
//
//for (auto n : material_counts) {
//    printf("%i, ", n);
//}
//
//// create object ptrs


//template <int n>
//std::array<Metal, n> build_metals(int& seed) {
//    std::array<Metal, n> metals{};
//
//    for (int i = 0; i < n; i++) {
//        auto& metal = metals[i];
//        metal.color = Float3::random(seed, 0.5, 1);
//        metal.roughness = random_float(seed, 0, 1);
//    }
//
//    return metals;
//}
//
//template <int n>
//std::array<Lambert, n> build_lamberts(int& seed) {
//    std::array<Lambert, n> lamberts{};
//    for (int i = 0; i < n; i++) {
//        auto& lambert = lamberts[i];
//        lambert.color = Float3::random(seed, 0.5, 1);
//    }
//    return lamberts;
//}
//
//template <int n>
//std::array<Dielectric, n> build_dielectrics(int& seed) {
//    std::array<Dielectric, n> dielectrics{};
//
//    for (int i = 0; i < n; i++) {
//        auto& dielectric = dielectrics[i];
//        dielectric.color = Float3::random(seed, 0.5, 1);
//        dielectric.refractive_index = random_float(seed, 0.5, 4);
//    }
//    return dielectrics;
//}