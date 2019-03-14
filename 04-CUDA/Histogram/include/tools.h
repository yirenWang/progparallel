#ifndef TOOLS_H
#define TOOLS_H

#include <stdint.h>
#include <tuple>
#include <string>
// #include <png++/png.hpp>
#include <random>


template <typename T>
void echo_matrix(T &mat, uint32_t width, uint32_t height);
std::tuple<uint32_t, uint32_t, std::string, std::string>  analyse_cli(std::string,  uint8_t* limage);
void load_image(uint32_t imageH, uint32_t imageW, std::string filename, uint8_t* limage);
void  generate_image(uint32_t imageH, uint32_t imageW, uint8_t* limage, float d, int g, float graine);
void save_image_8(uint32_t imageH, uint32_t imageW, uint32_t *h_OutputGPU, std::string outputFile);
void save_image_32(uint32_t imageH, uint32_t imageW, uint32_t *h_OutputGPU, std::string outputFile);
#endif
