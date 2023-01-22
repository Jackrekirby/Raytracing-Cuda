#pragma once
#include "tools.cuh"
#include "vec.cuh"
#include "ray.cuh"

class HitRecord {
public:
    GPU HitRecord() : p(Float3()), normal(Float3()), t(0), front_face(false) { }

    GPU void set_face_normal(const Ray& r, const Float3& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
    Float3 p;
    Float3 normal;
    float t;
    bool front_face;
};