#pragma once
#include "vec.cuh"
#include "ray.cuh"

constexpr float pi = 3.1415927F;

class Camera {
public:
    APUS Camera(
        Float3 origin, 
        Float3 lookat, 
        Float3 vup, 
        float vfov, // vertical field-of-view in degrees
        float aspect_ratio, 
        float aperture,
        float focus_dist
    ) {
        float theta = vfov * pi / 180.0F;
        float h = tan(theta / 2.0F);

        float viewport_height = 2.0F * h;
        float viewport_width = aspect_ratio * viewport_height;

        Float3 w = unit_vector(origin - lookat);
        Float3 u = unit_vector(cross(vup, w));
        Float3 v = cross(w, u);

        this->origin = origin;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2.0F - vertical / 2.0F - focus_dist * w;
        
        lens_radius = aperture / 2.0F;
    }

    GPU Ray compute_ray(float s, float t, int& seed) const {
        Float3 rd = lens_radius * random_in_unit_disk(seed);
        Float3 offset = u * rd.x + v * rd.y;

        return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    Float3 origin;
    Float3 horizontal;
    Float3 vertical;
    Float3 lower_left_corner;
    Float3 u, v, w;
    float lens_radius;
};

