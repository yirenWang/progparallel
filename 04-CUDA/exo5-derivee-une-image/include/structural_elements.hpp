#pragma once
#include <iostream>
#include <cmath>
#include "global_parameters.hpp"

void init_se_direction(u_char **StructuralElement, float angle, unsigned int height, unsigned int width);
void init_se_right(u_char **StructuralElement, unsigned int height, unsigned int width);
void init_se_left(u_char **StructuralElement, unsigned int height, unsigned int width);
void init_se_above(u_char **StructuralElement, unsigned int height, unsigned int width);
void init_se_under(u_char **StructuralElement, unsigned int height, unsigned int width);
