#include "config.hpp"

const std::uint32_t default_width = 16000; //2560; //16000; //5120;
const std::uint32_t default_height = 16000; //1440; //16000; //10000; //2880;
const std::uint32_t default_iterations = 500;

const float_type default_min_real = -1.2; //-1.9; // must be <= 0.0
const float_type default_max_real =  0.8; // 1.0; // must be >= 0.0
const float_type default_min_imag = -1.0; //-0.9; // must be <= 0.0
const float_type default_max_imag = default_min_imag
                                  + (default_max_real - default_min_real)
                                  * (static_cast<float_type>(default_height)
                                  / static_cast<float_type>(default_width));

const float_type default_scaling = 0.3;
