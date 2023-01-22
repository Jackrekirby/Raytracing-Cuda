#pragma once
#include "vec.cuh"
#include "hit_record.cuh"

enum class Material { Lambert, Metal, Dielectric };

class Lambert {
public:
    Lambert() : color(Float3()) {}
    Lambert(Float3 color) : color(color) {}
    Float3 color;
};

struct Metal {
public:
    Metal() : color(Float3()), roughness(0) {}
    Metal(Float3 color, float roughness) : color(color), roughness(roughness) {}
    Float3 color;
    float roughness;
};

struct Dielectric {
public:
    Dielectric() : color(Float3()), refractive_index(0) {}
    Dielectric(Float3 color, float refractive_index) : color(color), refractive_index(refractive_index) {}
    Float3 color;
    float refractive_index;
};

GPU bool scatter_lambert(const Ray& ray_in, const HitRecord& record, Ray& ray_out, int& seed) {
    Float3 direction = record.normal + random_unit_sphere(seed);

    // Catch degenerate scatter direction
    if (direction.near_zero())
        direction = record.normal;

    ray_out = Ray(record.p, direction);
    return true;
}

GPU bool scatter_metal(const Metal& metal, const Ray& ray_in, const HitRecord& record, Ray& ray_out, int& seed) {
    const Float3 direction = reflect(unit_vector(ray_in.direction), record.normal);

    ray_out = Ray(record.p, direction + metal.roughness * random_in_unit_sphere(seed));
    return (dot(ray_out.direction, record.normal) > 0);;
}

GPU bool scatter_dielectric(
    const Dielectric& dielectric, const Ray& ray_in, const HitRecord& record, Ray& ray_out, int& seed
) {
    float refraction_ratio = record.front_face ? (1.0 / dielectric.refractive_index) : dielectric.refractive_index;

    Float3 unit_direction = unit_vector(ray_in.direction);
    float cos_theta = fmin(dot(-unit_direction, record.normal), 1.0F);
    float sin_theta = sqrtf(1.0F - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;

    Float3 direction;

    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - refraction_ratio) / (1 + refraction_ratio);
    r0 = r0 * r0;
    auto reflectance = r0 + (1 - r0) * pow((1 - cos_theta), 5);

    if (cannot_refract || reflectance > random_float(seed))
        direction = reflect(unit_direction, record.normal);
    else
        direction = refract(unit_direction, record.normal, refraction_ratio);

    ray_out = Ray(record.p, direction);
    return true;
}