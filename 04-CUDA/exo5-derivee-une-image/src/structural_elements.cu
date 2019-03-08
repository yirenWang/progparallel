#include <structural_elements.hpp>

void init_se_direction(u_char **StructuralElement, float angle, unsigned int height, unsigned int width){

    float alpha = alpha = (angle * 3.14159) / 180.0;

    float u_alpha_r = std::sin(alpha);
	float u_alpha_c = std::cos(alpha);

    int r,c;
    int rb,cb;
    float fuzzy_val;

    for (auto r = 0; r<2*height; r++) {
        rb = height - r;
        for (auto c = 0; c<2*width; c++) {
            cb = c - width;
            if ((r == height) and (c == width)) fuzzy_val = 1.0;
            else fuzzy_val = std::max((float)0.0, static_cast<float>(1 -  ((float)FINENESS)*std::acos((rb*u_alpha_r + cb*u_alpha_c) / std::sqrt(rb*rb + cb*cb))));
            StructuralElement[r][c] = static_cast<u_char>(255*fuzzy_val);
            
        }
    }
}

void init_se_right(u_char **StructuralElement, unsigned int height, unsigned int width){
    init_se_direction(StructuralElement, 0.0, height, width);
}
void init_se_left(u_char **StructuralElement, unsigned int height, unsigned int width){
    init_se_direction(StructuralElement, 180.0, height, width);
}
void init_se_above(u_char **StructuralElement, unsigned int height, unsigned int width){
    init_se_direction(StructuralElement, 90.0, height, width);
}
void init_se_under(u_char **StructuralElement, unsigned int height, unsigned int width){
    init_se_direction(StructuralElement, -90.0, height, width);
}