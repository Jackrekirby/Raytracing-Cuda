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
    Objects(ObjectPtr* refs, Sphere* spheres, Lambert* lamberts, Metal* metals, Dielectric* dielectrics) :
        refs(refs), spheres(spheres), lamberts(lamberts), metals(metals), dielectrics(dielectrics) { }

    ObjectPtr* refs;
    Sphere* spheres;
    Lambert* lamberts;
    Metal* metals;
    Dielectric* dielectrics;
};