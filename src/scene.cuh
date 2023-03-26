#pragma once
#include <array>
#include <vector>
#include "materials.cuh"
#include "object.cuh"

constexpr uint n_materials = 4, n_shapes = 1;
class ObjectManager {
public:
    ObjectManager(uint n_objects, uint n_spheres, uint n_lamberts, uint n_metals, uint n_dielectrics, uint n_diffuse_lights):
        count_objects(0), total_objects(n_objects) {
        if (n_objects != n_spheres) {
            EXIT("n_objects != n_spheres");
        }
        if (n_objects != (n_lamberts + n_metals + n_dielectrics + n_diffuse_lights)) {
            EXIT("n_objects != (n_lamberts + n_metals + n_dielectrics + n_diffuse_lights)");
        }

        total_materials = { n_lamberts, n_metals, n_dielectrics, n_diffuse_lights };
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
        objects = Objects(objectPtrs.data(), spheres.data(), lamberts.data(), metals.data(), dielectrics.data(), diffuse_lights.data());
    }

    int material_index(Material material) {
        return static_cast<int>(material);
    }

    uint total_material(Material material) {
        return total_materials[material_index(material)];
    }

    uint count_material(Material material) {
        return count_materials[material_index(material)];
    }

    int get_material_left(Material material, bool diffuse_light_check = false) {
        if (
            diffuse_light_check && 
            material == Material::Dielectric && 
            total_material(Material::Dielectric) > total_material(Material::DiffuseLight)
        ) {
            return get_material_left(Material::Dielectric) - get_material_left(Material::DiffuseLight);
        }
        return total_material(material) - count_material(material);
    }

    void random_fill(
        int& seed,
        const Float3 offset,
        float min_placement_radius,
        float max_placement_radius,
        float min_radius,
        float max_radius,
        int max_attempts
    ) {
        while(count_objects < total_objects) {
            float r = random_float(seed);
            float r_check = 0;

            for (int i = 0; i < n_materials; ++i) {
                Material material = static_cast<Material>(i);
                const int material_left = get_material_left(material, true);
                const int objects_left = total_objects - count_objects;
                r_check += static_cast<float>(material_left) / static_cast<float>(objects_left);
                //printf("%i, %i, %i, %.4f\n", i, material_left, objects_left, r_check);
                
                if (r < r_check) {
                    bool was_made = false;
                    for (int j = 0; j < max_attempts; ++j) {
                        Sphere sphere = random_sphere(seed, offset, min_placement_radius, max_placement_radius, min_radius, max_radius);
                        was_made = add_sphere(sphere);
                        if (was_made) break;
                    }

                    if (!was_made) {
                        EXIT("Failed to place sphere");
                        return; // if failed to place sphere many times then give up
                    }

                    switch (material)
                    {
                    case Material::Lambert:
                        lamberts.push_back(random_lambert(seed));
                        add_object(Shape::sphere, material);
                        break;
                    case Material::Metal:
                        metals.push_back(random_metal(seed));
                        add_object(Shape::sphere, material);
                        break;
                    case Material::Dielectric:
                        dielectrics.push_back(random_dielectric(seed));
                        add_object(Shape::sphere, material);
                        break;
                    case Material::DiffuseLight:
                        const int dielectric_left = get_material_left(Material::Dielectric);
                        if (dielectric_left > 0) {
                            dielectrics.push_back({ Float3(1, 1, 1), 0.05f, 1.5f });
                            add_object(Shape::sphere, Material::Dielectric);

                            const Sphere& outer = spheres.back();
                            Sphere inner = { outer.center, outer.radius * 0.7f };
                            spheres.push_back(inner);
                        }
                        diffuse_lights.push_back(random_diffuse_light(seed));
                        add_object(Shape::sphere, material);
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
        float min_placement_radius,
        float max_placement_radius,
        float min_radius,
        float max_radius
    ) {
        
        float placement_radius = min_placement_radius + (max_placement_radius - min_placement_radius) * sqrt(random_float(seed));
        float radius = random_float(seed, min_radius, max_radius);
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
        const auto roughness = random_float(seed, 0, 0);
        return { color, roughness, refractive_index };
    }

    Lambert random_lambert(int& seed) {
        return { Float3::random(seed, 0.5, 1) };
    }

    Metal random_metal(int& seed) {
        const auto color = Float3::random(seed, 0.5, 1);
        const auto roughness = random_float(seed, 0, 1);
        return { color, roughness };
    }

    DiffuseLight random_diffuse_light(int& seed) {
        return { Float3::random(seed, 0.5, 1) };
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
    std::vector<DiffuseLight> diffuse_lights;
    Objects objects;
};