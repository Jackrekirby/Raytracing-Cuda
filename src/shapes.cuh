#pragma once
#include "tools.cuh"
#include "vec.cuh"
#include "ray.cuh"

enum class Shape { sphere };

class Sphere {
public:
    Sphere() : center(Float3()), radius(0) {}

    Sphere(const Float3& center, float radius)
        : center(center), radius(radius) {

    }

    GPU bool hit(const Ray& ray, float t_min, float t_max, HitRecord& record) const {
        Float3 oc = ray.origin - center;
        const float a = ray.direction.length_squared();
        const float half_b = dot(oc, ray.direction);
        const float c = oc.length_squared() - radius * radius;
        const float discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return false;
        const float sqrtd = sqrtf(discriminant);

        float root = (-half_b - sqrtd) / a;

        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root)
                return false;
        }

        record.t = root;
        record.p = ray.at(record.t);
        const Float3 outward_normal = (record.p - center) / radius;
        record.set_face_normal(ray, outward_normal);

        return true;
    }

    Float3 center;
    float radius;
};
