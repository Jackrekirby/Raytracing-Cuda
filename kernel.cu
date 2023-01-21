
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>

#include <array>

#include "tools.cuh"
#include "vec.cuh"
#include "ray.cuh"
#include "camera.cuh"
#include "image.cuh"
#include "timing.cuh"
#include "host_device_transfer.cuh"
#include "kernel_allocator.cuh"

enum class Shape { sphere };
enum class Material { Lambert, Metal, Dielectric };
typedef unsigned int uint;
constexpr uint n_spheres = 1000, n_lamberts = n_spheres, n_metals = n_spheres, n_dielectrics = n_spheres;
constexpr uint n_objects = n_spheres;

GPU void throw_error(const char* msg) {
    printf(msg);
    __trap();
}

struct ObjectPtr {
    ObjectPtr() : shape(Shape::sphere), shapeIndex(0), material(Material::Lambert), materialIndex(0) {
    }
    ObjectPtr(Shape shape, uint shapeIndex, Material material, uint materialIndex): shape(shape), shapeIndex(shapeIndex), 
        material(material), materialIndex(materialIndex) {
    }
    Shape shape;
    uint shapeIndex;
    Material material;
    uint materialIndex;
};

class HitRecord {
public:
    GPU HitRecord(): p(Float3()), normal(Float3()), t(0), front_face(false) { }

    GPU void set_face_normal(const Ray& r, const Float3& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
    Float3 p;
    Float3 normal;
    float t;
    bool front_face;
};

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

class Objects {
public:
    Objects(ObjectPtr* refs, Sphere* spheres, Lambert* lamberts, Metal* metals, Dielectric* dielectrics) :
        refs(refs), spheres(spheres), lamberts(lamberts), metals(metals), dielectrics(dielectrics) { }

    ObjectPtr* refs;
    Sphere* spheres;
    Lambert* lamberts;
    Metal* metals;
    Dielectric* dielectrics;
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


//constexpr float infinity = std::numeric_limits<float>::infinity();
//constexpr float pi = 3.1415927F;

GPU ObjectPtr* compute_hits(const Ray& ray, const Objects *objects, HitRecord &closest_hit_record) {
    float t_min = 0.001;
    float t_max = 3.40282e+038;
    
    HitRecord current_hit_record;
    ObjectPtr* closest_object = nullptr;

    for (int i = 0; i < n_objects; i++) {
        auto ref = &objects->refs[i];
        bool did_hit = false;

        switch (ref->shape)
        {
        case Shape::sphere:
            did_hit = objects->spheres[ref->shapeIndex].hit(ray, t_min, t_max, current_hit_record);
            break;
        }
       
        if (did_hit) {
            t_max = current_hit_record.t;
            closest_hit_record = current_hit_record;
            closest_object = ref;
        }
    }

    return closest_object;
}

GPU Float3 color_ray(const Ray& ray, const Objects* objects, int depth, int& seed) {
    Float3 color(0, 0, 0);
    Float3 total_attenuation(1, 1, 1);
    Ray ray_in = ray;

    for (int i = 0; i < depth; ++i) {
        HitRecord record;
        ObjectPtr* closest_object = compute_hits(ray_in, objects, record);
        if (closest_object != nullptr) {
            //return record.p;
            Ray ray_out;
            ObjectPtr &ref = *closest_object;
            bool doScatter = false;
            Float3 attenuation;

            switch (ref.material)
            {
            case Material::Lambert: {
                doScatter = scatter_lambert(ray_in, record, ray_out, seed);
                attenuation = objects->lamberts[ref.materialIndex].color;
                break;
            }
            case Material::Metal: {
                const Metal& metal = objects->metals[ref.materialIndex];
                doScatter = scatter_metal(metal, ray_in, record, ray_out, seed);
                attenuation = metal.color;
                break;
            }
            case Material::Dielectric: {
                const Dielectric& dielectric = objects->dielectrics[ref.materialIndex];
                doScatter = scatter_dielectric(dielectric, ray_in, record, ray_out, seed);
                attenuation = dielectric.color;
                break;
            }
            default:
                __trap();
                return Float3(1, 1, 0);
                //throw_error("Undefined Material");
            }
            
            if (!doScatter) return Float3(0, 0, 0);
            total_attenuation *= attenuation;
            
            ray_in = ray_out;
            if(i + 1 < depth) continue;
            
        }

        const Float3 unit_direction = unit_vector(ray_in.direction);
        const float t = 0.5 * (unit_direction.y + 1.0);

        
       
        color += total_attenuation * ((1.0F - t) * Float3(1, 1, 1) + t * Float3(0.5, 0.7, 1.0));
        break;
    }
    
    return color;
}



CUDA void compute_pixel(
    KernelAllocator* ka,
    Char3* pixels,
    Camera* camera,
    Objects* objects
) {
    int2 coords = ka->get_coords(threadIdx.x, blockIdx.x);
    const int x = coords.x;
    const int y = coords.y;
    const int width = ka->width;
    const int height = ka->height;

    if (y > height) return;

    const int i = x + y * width;
    int seed = i;

    const int samples_per_pixel = 30;
    const int max_depth = 10;

    Float3 color(0, 0, 0);
    for (int i = 0; i < samples_per_pixel; ++i) {
        const float u = float(x + random_float(seed)) / float(width - 1);
        const float v = float(height - 1 - y + random_float(seed)) / float(height - 1); // flip y axis

        const Ray ray = camera->compute_ray(u, v, seed);
      
        color += color_ray(ray, objects, max_depth, seed);
    }

    //const float r = float(x) / (width - 1);
    //const float g = float(y) / (height - 1);
    //const float b = 0.25;
    //Float3 color(r, g, b);

    const Char3 icolor = static_cast<Char3>(256.0F * clamp(sqrt(color / static_cast<float>(samples_per_pixel)), 0.0F, 0.999F));

    if (isnan(color.x)) {
        pixels[i] = Char3(0, 255, 0);
    } else if (!isfinite(color.x)) {
        pixels[i] = Char3(0, 0, 255);
    } else if (color.x == 0) {
        pixels[i] = Char3(255, 0, 0);
    }
    else {
        pixels[i] = icolor;
    }
}
    


void generate_scene(
    std::array<Sphere, n_spheres>& spheres, 
    float max_placement_radius, 
    float min_radius, 
    float max_radius, 
    int start_index,
    int& seed
) {

    int i = start_index;
    int k = 0;
    while(k < n_spheres * 10) {
        Sphere &sphere = spheres[i];
        float radius = random_float(seed, min_radius, max_radius);

        float placement_radius = max_placement_radius * sqrt(random_float(seed));
        float placement_angle = random_float(seed) * 2 * pi;

        Float3 center = Float3(
            placement_radius * cosf(placement_angle),
            radius, 
            placement_radius * sinf(placement_angle)
        );

        bool too_close = false;

        for (int j = 0; j < i; ++j) {
            Sphere other = spheres[j];

            float min_dist = radius + other.radius;
            if ((center - other.center).length_squared() < min_dist * min_dist) {
                too_close = true;
                break;
            }
        }

        if (!too_close) {
            sphere.center = center;
            sphere.radius = radius;
            ++i;
        } 

        if (i == n_spheres) break;
        k++;
    }
}


int render() {
    TimeIt t;
    t.start("render");

    int seed = 435735475; // 4357354765


    Image image(1920);

    Float3 lookfrom(5, 5, 5);
    Float3 lookat(0, 0, 0);
    Float3 vup(0, 1, 0);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 2.0F;
    float vfov = 80;
     
    Camera camera(lookfrom, lookat, vup, vfov, image.aspect_ratio, aperture, dist_to_focus);

    KernelAllocator ka(image.width, image.height);

    std::array<Lambert, n_lamberts> lamberts{};

    for (int i = 0; i < n_lamberts; i++) {
        auto& lambert = lamberts[i];
        lambert.color = Float3::random(seed, 0.5, 1);
    }

    std::array<Metal, n_metals> metals{};

    for (int i = 0; i < n_metals; i++) {
        auto& metal = metals[i];
        metal.color = Float3::random(seed, 0.5, 1);
        metal.roughness = random_float(seed, 0, 1);
    }

    std::array<Dielectric, n_dielectrics> dielectrics{};

    for (int i = 0; i < n_dielectrics; i++) {
        auto& dielectric = dielectrics[i];
        dielectric.color = Float3::random(seed, 0.5, 1);
        dielectric.refractive_index = random_float(seed, 0.5, 4);
    }

    std::array<Sphere, n_spheres> spheres = {
        Sphere(Float3(0,-1000,0), 1000)
    };

    generate_scene(spheres, 16.0F, 0.2F, 1.2F, 1, seed);

 /*   for (auto sphere : spheres) {
        printf("%.4f, %.4f, %.4f, %.4f\n", sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius);
    }*/

    std::array<ObjectPtr, n_objects> objectPtrs{};

    int lambert_index = 0;
    int metal_index = 0;
    int dielectric_index = 0;
    for (int i = 0; i < n_objects; i++) {
        auto& ref = objectPtrs[i];
        ref.shape = Shape::sphere;
        ref.shapeIndex = i;

        float r = random_float(seed);
        if (r < 0.5F) {
            ref.material = Material::Lambert;
            ref.materialIndex = lambert_index;
            lambert_index++;
        } else if (r < 0.75F) {
            ref.material = Material::Metal;
            ref.materialIndex = metal_index;
            metal_index++;
        }
        else {
            ref.material = Material::Dielectric;
            ref.materialIndex = dielectric_index;
            dielectric_index++;
        }
    }



    //Objects objects(lamberts.data());
    Objects objects(objectPtrs.data(), spheres.data(), lamberts.data(), metals.data(), dielectrics.data());

    GpuDataClone<Objects> c_objects(&objects);
    GpuDataClone<Sphere> c_spheres(spheres.data(), static_cast<int>(spheres.size()));
    GpuDataClone<Lambert> c_lamberts(lamberts.data(), static_cast<int>(lamberts.size()));
    GpuDataClone<Metal> c_metals(metals.data(), static_cast<int>(metals.size()));
    GpuDataClone<Dielectric> c_dielectrics(dielectrics.data(), static_cast<int>(dielectrics.size()));
    GpuDataClone<ObjectPtr> c_objectPtrs(objectPtrs.data(), static_cast<int>(objectPtrs.size()));

    GpuDataClone<Camera> c_camera(&camera);
    GpuDataClone<Char3> c_pixels(image.pixels.data(), static_cast<int>(image.pixels.size()));
    GpuDataClone<KernelAllocator> c_ka(&ka);
    
    VAR("height", ka.height);
    VAR("num_blocks", ka.num_blocks);
    VAR("num_threads", ka.num_threads);

    t.start("gpu allocation");
    cudaError_t cudaStatus;
    ON_ERROR_GOTO(cudaSetDevice(0));
    ON_ERROR_GOTO(c_pixels.allocate());
    ON_ERROR_GOTO(c_objects.allocate());
    ON_ERROR_GOTO(c_spheres.allocate());
    ON_ERROR_GOTO(c_lamberts.allocate());
    ON_ERROR_GOTO(c_metals.allocate());
    ON_ERROR_GOTO(c_dielectrics.allocate());
    ON_ERROR_GOTO(c_objectPtrs.allocate());
    ON_ERROR_GOTO(c_camera.allocate());
    ON_ERROR_GOTO(c_ka.allocate());
    t.stop();

    t.start("gpu data transfer");
    ON_ERROR_GOTO(c_objects.toGpu());
    ON_ERROR_GOTO(c_spheres.toGpu());
    ON_ERROR_GOTO(c_lamberts.toGpu());
    ON_ERROR_GOTO(c_metals.toGpu());
    ON_ERROR_GOTO(c_dielectrics.toGpu());
    ON_ERROR_GOTO(c_objectPtrs.toGpu());
    ON_ERROR_GOTO(c_camera.toGpu());
    ON_ERROR_GOTO(c_ka.toGpu());
    t.stop();

    copyDataToGpuBuffer(&c_objects.devPtr->spheres, &c_spheres.devPtr);
    copyDataToGpuBuffer(&c_objects.devPtr->lamberts, &c_lamberts.devPtr);
    copyDataToGpuBuffer(&c_objects.devPtr->metals, &c_metals.devPtr);
    copyDataToGpuBuffer(&c_objects.devPtr->dielectrics, &c_dielectrics.devPtr);
    copyDataToGpuBuffer(&c_objects.devPtr->refs, &c_objectPtrs.devPtr);

    t.start("compute_pixel");
    KERNEL(compute_pixel, ka.num_blocks, ka.num_threads)(
        c_ka.devPtr, 
        c_pixels.devPtr, 
        c_camera.devPtr, 
        c_objects.devPtr
    ); // max number of threads = 1024
    ON_ERROR_GOTO(cudaDeviceSynchronize());
    t.stop();
    
    t.start("get pixel data");
    ON_ERROR_GOTO(c_pixels.fromGpu());
    t.stop();

    t.start("save_image");
    image.save_as_bin("scene");
    t.stop();
ERROR:
    c_pixels.free();
    c_camera.free();
    c_ka.free();
    c_objects.free();
    c_spheres.free();
    c_lamberts.free();
    c_metals.free();
    c_dielectrics.free();
    c_objectPtrs.free();
    t.stop();
    ON_ERROR_RETURN(cudaStatus);
    return 0;
}

int main() {
    int runtimeVersion = 0;
    cudaError_t cudaStatus;
    ON_ERROR_RETURN(cudaRuntimeGetVersion(&runtimeVersion));
    VAR("CUDA VERSION", runtimeVersion);
    return render();    


    //for (int i = 0; i < 10000; ++i) {
    //    auto seed = i;
    //    auto val = Float3(random_float(seed, -1, 1), random_float(seed, -1, 1), 0);
    //    //test((val / val.length()).x);
    //    auto n = (val / val.length()).x;
    //    if (isnan(n) && n != 0.0F) {
    //        printf("%i, %.4f\n", i, n);
    //        printf("%.4f, %.4f, %.4f\n", val.x, val.y, val.z);
    //    }
    //}


}


