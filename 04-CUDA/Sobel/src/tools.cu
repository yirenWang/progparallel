#include "tools.h"

template <typename T>
void echo_matrix(T &mat, uint32_t width, uint32_t height)
{

	for (unsigned i = 0; i < width ; i++) {
		for (unsigned j = 0; j < height ; j++) {
			std::cout << *mat[i*width+j];
		}
		std::cout << std::endl;
	}
}

std::tuple<uint32_t, uint32_t, std::string, std::string> analyse_cli(std::string filename, uint8_t* limage) {

    auto idx = filename.find_last_of("/");
    std::string outputFile = filename;
    outputFile = outputFile.erase(0, idx + 1);

    // if (argc == 3) {
    //     inputFile = std::string(argv[1]);
    //     outputFile = std::string(argv[2]);
    // }

    png::image< png::rgb_pixel > image(filename);

    auto const imageW = image.get_width();
    auto const imageH = image.get_height();

    return std::make_tuple(imageH, imageW, filename, outputFile);
}

void  load_image(uint32_t imageH, uint32_t imageW, std::string filename, uint8_t* limage) {


    png::image< png::rgb_pixel > image(filename);
    for (auto i = 0; i < imageH; ++i)
     {
        for (auto j = 0; j < imageW; ++j)
         {
             limage[i*imageW + j] = image[i][j].red;
         }
     }
}

void  generate_image(uint32_t imageH, uint32_t imageW, uint8_t* limage, float d, int g, float graine) {
	std::mt19937 generator(graine);
	std::uniform_real_distribution<double> dist(0.0, 100.0);
    for (auto i = 0; i < imageH; i=i + g)
     {
        for (auto j = 0; j < imageW; j = j + g)
         {
			 double random_val = dist(generator);
			 for (auto k = 0; k < g; ++k) {
				 for (auto l = 0; l < g; ++l) {
					 if (((i+k) < imageH)&&((j+l) < imageW)) {
					 	if (random_val < d) {
		             		limage[(i+k)*imageW + j+l] = (uint8_t)255;
						}
						else limage[(i+k)*imageW + j+l] = (uint8_t)0;
					}
				}
			}
         }
     }
}

void save_image_8(uint32_t imageH, uint32_t imageW, uint32_t *h_OutputGPU, std::string outputFile) {

    png::image< png::rgb_pixel > image(imageW, imageH);

    for (auto i = 0; i < imageH; ++i)
    {
        for (auto j = 0; j < imageW; ++j)
        {
            auto val = h_OutputGPU[i*imageW + j];
            image[i][j] = png::rgb_pixel(val, val, val);
        }
    }

    image.write(outputFile);
}

void save_image_32(uint32_t imageH, uint32_t imageW, uint32_t *h_OutputGPU, std::string outputFile) {

    png::image< png::rgb_pixel > image(imageW, imageH);

    for (auto i = 0; i < imageH; ++i)
    {
        for (auto j = 0; j < imageW; ++j)
        {
            auto valr = 123*h_OutputGPU[i*imageW + j];
            auto valg = 321*(h_OutputGPU[i*imageW + j]&0x0000ff00)>>8;
            auto valb = 021*(h_OutputGPU[i*imageW + j]&0x00ff0000)>>16;
            // auto valr = h_OutputGPU[y*imageW + x];
            // auto valg = (h_OutputGPU[y*imageW + x]&0x0000ff00)>>8;
            // auto valb = (h_OutputGPU[y*imageW + x]&0x00ff0000)>>16;

            image[i][j] = png::rgb_pixel(valr, valg, valb);
        }
    }

    image.write(outputFile);
}
