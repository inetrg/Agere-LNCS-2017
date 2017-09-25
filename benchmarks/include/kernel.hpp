#ifndef KERNEL_HPP
#define KERNEL_HPP

namespace {
constexpr const char* kernel_name  = "matrix_mult";
constexpr const char* kernel_name2 = "l_dim_1";
constexpr const char* kernel_name3 = "l_dim_2";
constexpr const char* kernel_name4 = "matrix_mult_int";
constexpr const char* kernel_name5 = "matrix_sqr";
constexpr const char* kernel_name6 = "cpy";
constexpr const char* kernel_name7 = "cpy_more";
constexpr const char* kernel_name8 = "cpy_3d";

constexpr const char* kernel_source = R"__(
    __kernel void matrix_mult(__global float* matrix1,
                              __global float* matrix2,
                              __global float* output) {
        size_t size = get_global_size(0); // == get_global_size_(1);
        size_t x = get_global_id(0);
        size_t y = get_global_id(1);
        float result = 0;
        for (size_t idx = 0; idx < size; ++idx) {
            result += matrix1[idx + y * size] * matrix2[x + idx * size];
        }
        output[x+y*size] = result;
    }

    __kernel void matrix_mult_int(__global int* matrix1,
                                  __global int* matrix2,
                                  __global int* output) {
        size_t size = get_global_size(0); // == get_global_size_(1);
        size_t x = get_global_id(0);
        size_t y = get_global_id(1);
        int result = 0;
        for (size_t idx = 0; idx < size; ++idx) {
            result += matrix1[idx + y * size] * matrix2[x + idx * size];
        }
        output[x+y*size] = result;
    }

    __kernel void l_dim_1(__global float* input,
                          __global float* output) {
        size_t l_size = get_local_size(0);
        size_t l_id = get_local_id(0);
        size_t pos = get_global_id(0);
        if(l_id == 0) {
            output[pos] = l_size;
        }
    }

    __kernel void l_dim_2(__global float* input,
                          __global float* output) {
        size_t g_size = get_global_size(0);
        size_t l_size_0 = get_local_size(0);
        size_t l_size_1 = get_local_size(1);
        size_t l_id_0 = get_local_id(0);
        size_t l_id_1 = get_local_id(1);
        size_t x = get_global_id(0);
        size_t y = get_global_id(1);
        if(l_id_0 == 0 && l_id_1 == 0) {
            output[y * g_size + x] = l_size_0;
        }
        else if(l_id_0 == 0 && l_id_1 == 1) {
            output[y * g_size + x] = l_size_1;
        }
        else if(l_id_0 == 1 && l_id_1 == 0) {
            output[y * g_size + x] = l_size_1;
        }
    }

    __kernel void matrix_sqr(__global float* matrix,
                             __global float* output) {
        size_t size = get_global_size(0); // == get_global_size_(1);
        size_t x = get_global_id(0);
        size_t y = get_global_id(1);
        float result = 0;
        for (size_t idx = 0; idx < size; ++idx) {
            result += matrix[idx + y * size] * matrix[x + idx * size];
        }
        output[x+y*size] = result;
    }

    __kernel void cpy(__global float* input, __global float* output) {
        size_t idx = get_global_id(0);
        output[idx] = input[idx];
    }

    __kernel void cpy_more(__global float* input, __global float* output) {
        size_t size = get_global_size(0); // == get_global_size_(1);
        size_t x = get_global_id(0);
        if (x == 0) {
            for (size_t i = 0; i < size; ++i) {
                output[i] = input[i];
            }
        }
    }

    __kernel void cpy_3d(__global float* input, __global float* output) {
        size_t x = get_global_id(0);
        size_t y = get_global_id(1);
        size_t z = get_global_id(2);
        size_t size = get_global_size(0); // x == y == z
        output[x + size * (y + size * z)] = input[x + size * (y + size * z)];
    }

)__";

} // namespace <anonymous>

#endif // KERNEL_HPP
