
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
#include "src/scene.cuh"

//constexpr uint n_spheres = 3, n_lamberts = 0, n_metals = 1, n_dielectrics = 1, n_diffuse_lights = 1;
constexpr uint n_spheres = 1000, n_lamberts = 200, n_metals = 400, n_dielectrics = 400, n_diffuse_lights = 0;
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
    Float3 emission(0, 0, 0);
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
            case Material::DiffuseLight: {
                const DiffuseLight& light = objects->diffuse_lights[ref.materialIndex];
                doScatter = false;
                emission = light.color;
                break;
            }
            default:
                CUDA_EXIT("Undefined Material");
            }
            
            if (!doScatter) return total_attenuation * emission;
            total_attenuation *= attenuation;
            
            ray_in = ray_out;
            if(i + 1 < depth) continue; 
        }

        const Float3 unit_direction = unit_vector(ray_in.direction);
        const Float3 light_direction = Float3(1, 1, 1);
        const float directional_light = dot(unit_vector(light_direction), unit_direction);
        const float ambient_light = 0.8f;
        float skylight = ambient_light > directional_light ? ambient_light : directional_light;

        ObjectPtr* closest_object2 = compute_hits(Ray(ray_in.origin, light_direction), objects, record);
        if (closest_object2 == nullptr) {
            skylight = 1.0f;
        }
        else {
            skylight = 0.0f;
        }

        const int skybox_pixel_index = coordinate_to_skybox_pixel_index(unit_direction, skybox_face_length);
        const Float3 skycolor = static_cast<Float3>(skybox_pixels[skybox_pixel_index]) / 256.0F;
        color += total_attenuation * skycolor;
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

    const int samples_per_pixel = 30;
    const int max_depth = 10;

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
    

int render() {
    TimeIt t;
    t.start("render");

    int seed = 435735475; // 4357354765
    std::stringstream ss;
    
    // 2360, 1640
    Image image(1920);

    Float3 lookfrom(13, 2, 3);
    Float3 lookat(0, 0, 0);
    Float3 vup(0, 1, 0);
    float dist_to_focus = 10.0f; //(lookfrom - lookat).length();
    float aperture = 0.1F;
    float vfov = 20;
     
    Camera camera(lookfrom, lookat, vup, vfov, image.aspect_ratio, aperture, dist_to_focus);

    KernelAllocator ka(image.width, image.height);

    ObjectManager obj_mgr(n_objects, n_spheres, n_lamberts, n_metals, n_dielectrics, n_diffuse_lights);
    obj_mgr.add_object(Shape::sphere, Material::Metal);
    obj_mgr.spheres.push_back({ Float3(0,-1000.0f,0), 1000.0f });
    obj_mgr.metals.push_back({ Metal(Float3(0.7f, 0.7f, 1.0f), 0.4f) });

    obj_mgr.add_object(Shape::sphere, Material::Dielectric);
    obj_mgr.spheres.push_back({ Float3(0, 1, 0), 1.0f });
    obj_mgr.dielectrics.push_back({ Dielectric(Float3(1, 1, 1), 0.0f, 1.5f) });

    obj_mgr.add_object(Shape::sphere, Material::Lambert);
    obj_mgr.spheres.push_back({ Float3(-4, 1, 0), 1.0f });
    obj_mgr.lamberts.push_back({ Lambert(Float3(0.7f, 1.0f, 0.4f)) });

    obj_mgr.add_object(Shape::sphere, Material::Metal);
    obj_mgr.spheres.push_back({ Float3(4, 1, 0), 1.0f });
    obj_mgr.metals.push_back({ Metal(Float3(0.7, 0.6, 0.5), 0.0f) });

    obj_mgr.random_fill(seed, Float3(0, 0, 0), 0.0F, 11.0F, 0.1F, 0.3F, 100);
    obj_mgr.build_objects();

    std::vector<Char3> skybox_pixels = import_skybox(skybox_face_length);

    AlienManager am;
    auto &c_objects = am.add(obj_mgr.objects);
    auto &c_spheres = am.add(obj_mgr.spheres);
    auto &c_lamberts = am.add(obj_mgr.lamberts);
    auto &c_metals = am.add(obj_mgr.metals);
    auto &c_dielectrics = am.add(obj_mgr.dielectrics);
    auto& c_diffuse_lights = am.add(obj_mgr.diffuse_lights);
    auto &c_objectPtrs = am.add(obj_mgr.objectPtrs);
    auto &c_camera = am.add(camera);
    auto &c_pixels = am.add(image.pixels, AlienType::OUT);
    auto &c_skybox_pixels = am.add(skybox_pixels);
    auto &c_ka = am.add(ka);
    
    VAR("height", ka.height);
    VAR("num_blocks", ka.num_blocks);
    VAR("num_threads", ka.num_threads);

    t.start("gpu allocation");
    cudaError_t cudaStatus;
    ON_ERROR_GOTO(cudaSetDevice(0));
    ON_ERROR_GOTO(am.allocate());
    
    t.stop();

    t.start("gpu data transfer");
    ON_ERROR_GOTO(am.toGpu());
    t.stop();

    copyDataToGpuBuffer(&c_objects.devPtr->spheres, &c_spheres.devPtr);
    copyDataToGpuBuffer(&c_objects.devPtr->lamberts, &c_lamberts.devPtr);
    copyDataToGpuBuffer(&c_objects.devPtr->metals, &c_metals.devPtr);
    copyDataToGpuBuffer(&c_objects.devPtr->dielectrics, &c_dielectrics.devPtr);
    copyDataToGpuBuffer(&c_objects.devPtr->diffuse_lights, &c_diffuse_lights.devPtr);
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
    ON_ERROR_GOTO(am.fromGpu());
    t.stop();

    t.start("save_image");
    image.save_as_bin("scene");
    t.stop();

    t.start("bin2png");
    ss << "node ./bin2png/main.js scene img/scene " << image.width << " " << image.height;
    system(ss.str().c_str());
    t.stop();
ERROR:
    am.free();
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
}

