﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <sstream>
#include <array>

#include "src/tools.cuh"
#include "src/vec.cuh"
#include "src/ray.cuh"
#include "src/camera.cuh"
#include "src/image.cuh"
#include "src/timing.cuh"
#include "src/host_device_transfer.cuh"
#include "src/kernel_allocator.cuh"
#include "src/materials.cuh"
#include "src/hit_record.cuh"
#include "src/shapes.cuh"
#include "src/object.cuh"
#include "src/skybox.cuh"

constexpr uint n_spheres = 1000, n_lamberts = n_spheres, n_metals = n_spheres, n_dielectrics = n_spheres;
constexpr uint n_objects = n_spheres;
constexpr int skybox_face_length = 900;

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

GPU Float3 color_ray(const Ray& ray, const Objects* objects, int depth, const Char3* skybox_pixels, int& seed) {
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

        const int skybox_pixel_index = coordinate_to_skybox_pixel_index(unit_direction, skybox_face_length);
        color += total_attenuation * static_cast<Float3>(skybox_pixels[skybox_pixel_index]) / 256.0F;
        break;
    }
    
    return color;
}

CUDA void compute_pixel(
    const KernelAllocator* ka,
    Char3* pixels,
    const Camera* camera,
    const Objects* objects,
    const Char3* skybox_pixels
) {
    int2 coords = ka->get_coords(threadIdx.x, blockIdx.x);
    const int x = coords.x;
    const int y = coords.y;
    const int width = ka->width;
    const int height = ka->height;

    if (y > height) return;

    const int i = x + y * width;
    int seed = i;

    const int samples_per_pixel = 1;
    const int max_depth = 1;

    Float3 color(0, 0, 0);
    for (int i = 0; i < samples_per_pixel; ++i) {
        const float u = float(x + random_float(seed)) / float(width - 1);
        const float v = float(height - 1 - y + random_float(seed)) / float(height - 1); // flip y axis

        const Ray ray = camera->compute_ray(u, v, seed);
      
        color += color_ray(ray, objects, max_depth, skybox_pixels, seed);
    }

    const Char3 icolor = static_cast<Char3>(256.0F * clamp(sqrt(color / static_cast<float>(samples_per_pixel)), 0.0F, 0.999F));

    if (isnan(color.x)) {
        pixels[i] = Char3(255, 0, 0);
    } else if (!isfinite(color.x)) {
        pixels[i] = Char3(0, 255, 0);
    } 
    else {
        pixels[i] = icolor;
    }
}
    


void generate_scene(
    std::array<Sphere, n_spheres>& spheres, 
    const Float3 offset,
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
        ) + offset;

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
    std::stringstream ss;

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

    std::array<Metal, n_metals> metals{
        //Metal(Float3(1, 1, 1), 0)
    };

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
        //Sphere(Float3(0,0,0), 3)
    };

    generate_scene(spheres, Float3(0, 0, 0), 16.0F, 0.2F, 1.2F, 1, seed);

 /*   for (auto sphere : spheres) {
        printf("%.4f, %.4f, %.4f, %.4f\n", sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius);
    }*/

    std::array<ObjectPtr, n_objects> objectPtrs{ ObjectPtr(Shape::sphere, 0, Material::Metal, 0)};

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

    std::vector<Char3> skybox_pixels = import_skybox(skybox_face_length);

    //Objects objects(lamberts.data());
    Objects objects(objectPtrs.data(), spheres.data(), lamberts.data(), metals.data(), dielectrics.data());

    GpuDataClone<Objects> c_objects(&objects);
    GpuDataClone<Sphere> c_spheres(spheres);
    GpuDataClone<Lambert> c_lamberts(lamberts);
    GpuDataClone<Metal> c_metals(metals);
    GpuDataClone<Dielectric> c_dielectrics(dielectrics);
    GpuDataClone<ObjectPtr> c_objectPtrs(objectPtrs);

    GpuDataClone<Camera> c_camera(&camera);
    GpuDataClone<Char3> c_pixels(image.pixels);
    GpuDataClone<Char3> c_skybox_pixels(skybox_pixels);
    GpuDataClone<KernelAllocator> c_ka(&ka);
    
    VAR("height", ka.height);
    VAR("num_blocks", ka.num_blocks);
    VAR("num_threads", ka.num_threads);

    t.start("gpu allocation");
    cudaError_t cudaStatus;
    ON_ERROR_GOTO(cudaSetDevice(0));
    ON_ERROR_GOTO(c_pixels.allocate());
    ON_ERROR_GOTO(c_skybox_pixels.allocate());
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
    ON_ERROR_GOTO(c_skybox_pixels.toGpu());
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
        c_objects.devPtr,
        c_skybox_pixels.devPtr
    ); // max number of threads = 1024
    ON_ERROR_GOTO(cudaDeviceSynchronize());
    t.stop();
    
    t.start("get pixel data");
    ON_ERROR_GOTO(c_pixels.fromGpu());
    t.stop();

    t.start("save_image");
    image.save_as_bin("scene");
    t.stop();

    t.start("bin2png");
    ss << "node ./bin2png/main.js scene img/scene " << image.width << " " << image.height;
    system(ss.str().c_str());
    t.stop();
ERROR:
    c_pixels.free();
    c_skybox_pixels.free();
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

void test() {
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
        int n = static_cast<float>(n_objects) * static_cast<float>(material_propensities[i]) / static_cast<float>(max_propensity);
        material_counts[i] = n;
        current_total_objects += n;
    }
    material_counts[0] = n_objects - current_total_objects;

    // create object ptrs
    std::array<ObjectPtr, n_objects> objectPtrs{};

    int objects_remaining = n_objects;
    for (int i = 0; i < n_objects; ++i) {
        float r = random_float(seed);
        int r_check = 0;
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

int main() {
    int runtimeVersion = 0;
    cudaError_t cudaStatus;
    ON_ERROR_RETURN(cudaRuntimeGetVersion(&runtimeVersion));
    VAR("CUDA VERSION", runtimeVersion);
    return render();    
}


