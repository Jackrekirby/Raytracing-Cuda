#pragma once
#include "materials.cuh"
#include "shapes.cuh"

class ObjectPtr {
public:
    ObjectPtr() : shape(Shape::sphere), shapeIndex(0), material(Material::Lambert), materialIndex(0) {
    }
    ObjectPtr(Shape shape, uint shapeIndex, Material material, uint materialIndex) : shape(shape), shapeIndex(shapeIndex),
        material(material), materialIndex(materialIndex) {
    }
    Shape shape;
    uint shapeIndex;
    Material material;
    uint materialIndex;
};

class Objects {
public:
    Objects(): refs(nullptr), spheres(nullptr), lamberts(nullptr), metals(nullptr), dielectrics(nullptr), diffuse_lights(nullptr) {}

    Objects(ObjectPtr* refs, Sphere* spheres, Lambert* lamberts, Metal* metals, Dielectric* dielectrics, DiffuseLight* diffuse_lights) :
        refs(refs), spheres(spheres), lamberts(lamberts), metals(metals), dielectrics(dielectrics), diffuse_lights(diffuse_lights) { }

    ObjectPtr* refs;
    Sphere* spheres;
    Lambert* lamberts;
    Metal* metals;
    Dielectric* dielectrics;
    DiffuseLight* diffuse_lights;
};