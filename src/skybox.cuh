GPU int coordinate_to_skybox_pixel_index(const Float3& coordinate, const int skybox_face_length)
{
    Float3 abs_c = abs(coordinate);
    Bool3 is_positive = coordinate.is_positive();

    float max_axis, uc, vc;
    int face_index = 0;

    if (abs_c.x >= abs_c.y && abs_c.x >= abs_c.z) {
        max_axis = abs_c.x;
        vc = -coordinate.y;
        if (is_positive.x) {
            uc = -coordinate.z;
            face_index = 1;
        }
        else {
            uc = coordinate.z;
            face_index = 0;
        }
    }

    if (abs_c.y >= abs_c.x && abs_c.y >= abs_c.z) {
        max_axis = abs_c.y;
        uc = -coordinate.x;
        if (is_positive.y) {
            vc = -coordinate.z;
            face_index = 2;
        }
        else {
            vc = coordinate.z;
            face_index = 3;
        }
    }

    if (abs_c.z >= abs_c.x && abs_c.z >= abs_c.y) {
        max_axis = abs_c.z;
        vc = -coordinate.y;
        if (is_positive.z) {
            uc = coordinate.x;
            face_index = 5;
        }
        else {
            uc = -coordinate.x;
            face_index = 4;
        }
    }

    // Convert range from -1 to 1 to 0 to 1
    float u = 0.5f * (uc / max_axis + 1.0f);
    float v = 0.5f * (vc / max_axis + 1.0f);

    float f_size = static_cast<float>(skybox_face_length);
    int x = static_cast<int>(u * f_size);
    int y = static_cast<int>(v * f_size);
    int pixel_index = x + y * skybox_face_length + face_index * skybox_face_length * skybox_face_length;
    return pixel_index;
}

GPU Float3 calculate_sky_color(const Float3 &unit_direction) {
    const float t = 0.5 * (unit_direction.y + 1.0);
    const Float3 sky_color = ((1.0F - t) * Float3(1, 1, 1) + t * Float3(0.5, 0.7, 1.0));
    return sky_color;
}


CPU std::vector<Char3> import_skybox(int face_length) {
    std::vector<Char3> pixels(face_length * face_length * 6);
    std::ifstream file("skybox/skybox.bin", std::ios::binary);
    file.read((char*)pixels.data(), sizeof(Char3) * pixels.size());
    file.close();
    return pixels;
}