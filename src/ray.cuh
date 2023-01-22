#pragma once
#include "vec.cuh"

class Ray {
public:
    GPU Ray() {}
    GPU Ray(const Float3& origin, const Float3& direction)
        : origin(origin), direction(direction)
    {}

    GPU Float3 at(float t) const {
        return origin + t * direction;
    }

    Float3 origin;
    Float3 direction;
};
