#pragma once
#include "tools.cuh"
#include <cmath>

typedef unsigned int uint;

template <typename T>
APUS T clamp(T x, T min, T max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

APUS int xorshift(int value) {
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
}

APUS float random_float(int& seed) {
    seed = xorshift(seed);
    const float x = float(seed) / 3141.592653F;
    return x - floorf(x);
}

APUS float random_float(int& seed, float min, float max) {
    return min + random_float(seed) * (max - min);
}

struct Bool3 {
    bool x, y, z;
};

template <typename T>
class vec3 {
public:
    APUS vec3() : x(0), y(0), z(0) {}

    APUS static vec3<T> tri(T w) {
        return vec3<T>(w, w, w);
    }

    APUS static vec3<float> random(int& seed) {
        return vec3<float>(
            random_float(seed),
            random_float(seed),
            random_float(seed)
        );
    }

    APUS static vec3<float> random(int &seed, float min, float max) {
        return vec3<float>(
            random_float(seed, min, max),
            random_float(seed, min, max),
            random_float(seed, min, max)
        );
    }

    APUS vec3(T x, T y, T z) : x(x), y(y), z(z) {

    }

    template <typename S>
    APUS operator vec3<S>() const {
        return vec3<S>(static_cast<S>(x), static_cast<S>(y), static_cast<S>(z));
    }

    APUS vec3 operator-() const { return vec3(-x, -y, -z); }

    APUS vec3& operator+=(const vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    APUS vec3& operator*=(const vec3& v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    APUS vec3& operator*=(const T t) {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    APUS T length() const {
        return sqrtf(length_squared());
    }

    APUS T length_squared() const {
        return x * x + y * y + z * z;
    }

    APUS Bool3 is_positive() const {
        return {x > 0, y > 0, z > 0};
    }

    APUS bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        const float s = 1e-8;
        return (fabs(x) < s) && (fabs(y) < s) && (fabs(z) < s);
    }

    T x, y, z;
};

template <typename T>
APUS bool operator==(const vec3<T>& v, const vec3<T>& u) {
    return v.x == u.x && v.y == u.y && v.z == u.z;
}

template <typename T>
APUS vec3<T> operator*(T t, const vec3<T>& v) {
    return vec3<T>(t * v.x, t * v.y, t * v.z);
}

template <typename T>
APUS vec3<T> operator*(const vec3<T>& v, T t) {
    return t * v;
}

template <typename T>
APUS vec3<T> operator*(const vec3<T>& u, const vec3<T>& v) {
    return vec3<T>(u.x * v.x, u.y * v.y, u.z * v.z);
}

template <typename T>
APUS vec3<T> operator+(const vec3<T>& u, const vec3<T>& v) {
    return vec3<T>(u.x + v.x, u.y + v.y, u.z + v.z);
}

template <typename T>
APUS vec3<T> operator-(const vec3<T>& u, const vec3<T>& v) {
    return vec3<T>(u.x - v.x, u.y - v.y, u.z - v.z);
}

template <typename T>
APUS vec3<T> operator/(vec3<T> v, T t) {
    return (1 / t) * v;
}

template <typename T>
APUS vec3<T> unit_vector(vec3<T> v) {
    auto x = v.length();
    if(x < 1e-8) {
        return Float3();
    }
    else {
        return v / v.length();
    }
}

template <typename T>
APUS vec3<T> clamp(vec3<T> v, T min, T max) {
    return vec3<T>(clamp(v.x, min, max), clamp(v.y, min, max), clamp(v.z, min, max));
}

APUS vec3<float> sqrt(vec3<float> v) {
    return vec3<float>(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

APUS vec3<float> abs(vec3<float> v) {
    return vec3<float>(abs(v.x), abs(v.y), abs(v.z));
}


template <typename T>
APUS T dot(const vec3<T>& u, const vec3<T>& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

typedef vec3<float> Float3;
typedef vec3<int> Int3;
typedef vec3<unsigned char> Char3;

GPU Float3 random_unit_sphere(int& seed) {
    return unit_vector(Float3::random(seed, -1, 1));
}

GPU Float3 random_unit_disk(int& seed) {
    return unit_vector(Float3(random_float(seed, -1, 1), random_float(seed, -1, 1), 0));
}

GPU Float3 random_in_unit_sphere(int& seed) {
    return random_unit_sphere(seed) * Float3::random(seed);
}

GPU Float3 random_in_unit_disk(int& seed) {
    return random_unit_disk(seed) * Float3::random(seed);
}

GPU Float3 random_in_hemisphere(int& seed, const Float3& normal) {
    Float3 r = random_in_unit_sphere(seed);
    return dot(r, normal) > 0 ? r : -r;
}

GPU Float3 refract(const Float3& uv, const Float3& n, float etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0F);
    Float3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Float3 r_out_parallel = -sqrt(fabs(1.0F - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

template <typename T>
GPU vec3<T> reflect(const vec3<T>& v, const vec3<T>& n) {
    return v - 2 * dot(v, n) * n;
}

template <typename T>
APUS vec3<T> cross(const vec3<T>& u, const vec3<T>& v) {
    return vec3<T>(u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x);
}


//GPU float rand(float x, float y) {
//    const float z = sinf(x * 12.9898F + y * 78.233F) * 43758.5453F;
//    const float fract = z - floorf(z);
//    return fract;
//}
//



//class vec3 {
//public:
//    vec3() : e{ 0,0,0 } {}
//    vec3(double e0, double e1, double e2) : e{ e0, e1, e2 } {}
//
//    static vec3 zero() {
//        return vec3(0, 0, 0);
//    }
//
//    static vec3 one() {
//        return vec3(1, 1, 1);
//    }
//
//    double x() const { return e[0]; }
//    double y() const { return e[1]; }
//    double z() const { return e[2]; }
//
//    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
//    double operator[](int i) const { return e[i]; }
//    double& operator[](int i) { return e[i]; }
//
//    vec3& operator+=(const vec3& v) {
//        e[0] += v.e[0];
//        e[1] += v.e[1];
//        e[2] += v.e[2];
//        return *this;
//    }
//
//    vec3& operator*=(const double t) {
//        e[0] *= t;
//        e[1] *= t;
//        e[2] *= t;
//        return *this;
//    }
//
//    vec3& operator/=(const double t) {
//        return *this *= 1 / t;
//    }
//
//    double length() const {
//        return std::sqrt(length_squared());
//    }
//
//    double length_squared() const {
//        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
//    }
//
//public:
//    double e[3];
//};
//
//
//inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
//    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
//}
//
//inline vec3 operator+(const vec3& u, const vec3& v) {
//    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
//}
//
//inline vec3 operator-(const vec3& u, const vec3& v) {
//    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
//}
//
//inline vec3 operator*(const vec3& u, const vec3& v) {
//    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
//}
//
//inline vec3 operator*(double t, const vec3& v) {
//    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
//}
//
//inline vec3 operator*(const vec3& v, double t) {
//    return t * v;
//}
//
//inline vec3 operator/(vec3 v, double t) {
//    return (1 / t) * v;
//}
//
//inline double dot(const vec3& u, const vec3& v) {
//    return u.e[0] * v.e[0]
//        + u.e[1] * v.e[1]
//        + u.e[2] * v.e[2];
//}
//
//inline vec3 cross(const vec3& u, const vec3& v) {
//    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
//        u.e[2] * v.e[0] - u.e[0] * v.e[2],
//        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
//}
//
//inline vec3 unit_vector(vec3 v) {
//    return v / v.length();
//}



