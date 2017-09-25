
#include <array>
#include <vector>
#include <future>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include "caf/all.hpp"

#include "include/config.hpp"

using namespace std;
using namespace caf;

namespace {

using matrix_type = vector<float>;

inline float& get(matrix_type& m, size_t size, size_t col, size_t row) {
  return m[size * row + col];
}

inline const float& get(const matrix_type& m, size_t size, size_t col, size_t row) {
  return m[size * row + col];
}

int dot_product(const matrix_type& lhs, const matrix_type& rhs,
                size_t size, size_t column, size_t row) {
  int result = 0.0f;
  for (size_t k = 0; k < size; ++k)
    result += get(lhs, size, k, row) * get(rhs, size, column, k);
  return result;
}

matrix_type simple_multiply(const matrix_type& lhs, const matrix_type& rhs, size_t size) {
  matrix_type result(size * size);
  for (size_t row = 0; row < size; ++row)
    for (size_t column = 0; column < size; ++column)
      get(result, size, column, row) = dot_product(lhs, rhs, size, column, row);
  return result;
}

matrix_type actor_multiply(actor_system& sys,
                           const matrix_type& lhs, const matrix_type& rhs,
                           size_t size) {
  matrix_type result(size * size);
  for (size_t row = 0; row < size; ++row)
    for (size_t column = 0; column < size; ++column)
      sys.spawn([&,row,column, size] {
        get(result, size, column, row) = dot_product(lhs, rhs, size, column, row);
      });
  sys.await_all_actors_done();
  return result;
}

matrix_type actor_multiply2(actor_system& sys,
                           const matrix_type& lhs, const matrix_type& rhs,
                           size_t size) {
  matrix_type result(size * size);
  for (size_t row = 0; row < size; ++row)
    sys.spawn([&,row, size] {
      for (size_t column = 0; column < size; ++column)
        get(result, size, column, row) = dot_product(lhs, rhs, size, column, row);
    });
  sys.await_all_actors_done();
  return result;
}

matrix_type async_multiply(const matrix_type& lhs, const matrix_type& rhs, size_t size) {
  matrix_type result(size * size);
  vector<future<void>> futures;
  futures.reserve(size * size);
  for (size_t row = 0; row < size; ++row) {
    for (size_t column = 0; column < size; ++column) {
      futures.push_back(std::async(std::launch::async, [&,row,column, size] {
        get(result, size, column, row) = dot_product(lhs, rhs, size, column, row);
      }));
    }
  }
  for (auto& f : futures)
    f.wait();
  return result;
}

matrix_type async_multiply2(const matrix_type& lhs, const matrix_type& rhs, size_t size) {
  matrix_type result(size * size);
  vector<future<void>> futures;
  futures.reserve(size);
  for (size_t row = 0; row < size; ++row)
    futures.push_back(std::async(std::launch::async, [&,row, size] {
      for (size_t column = 0; column < size; ++column)
        get(result, size, column, row) = dot_product(lhs, rhs, size, column, row);
    }));
  for (auto& f : futures)
    f.wait();
  return result;
}

matrix_type create_matrix(size_t size) {
    matrix_type matrix(size * size);
    iota(begin(matrix), end(matrix), 0);
    return matrix;
}


class config : public actor_system_config {
public:
  size_t size = 0;
  //  announce<vector<float>>("vector_float");
  config() {
    opt_group{custom_options_, "global"}
    .add(size, "size,s", "set matrix size (must be > 0)");
  }
};

void caf_main(actor_system& system, const config& cfg) {
  auto matrix_size = cfg.size;

  matrix_type m1 = create_matrix(matrix_size);
  matrix_type m2 = create_matrix(matrix_size);

  matrix_type matrix = actor_multiply2(system, m1, m2, matrix_size);

#ifdef CL_ENABLE_DEBUG
  for (size_t column = 0; column < matrix_size; ++column) {
    for (size_t row = 0; row < matrix_size; ++row) {
      cout << fixed << setprecision(2) << setw(9)
           << matrix[row + column * matrix_size];
    }
    cout << endl;
  }
#endif
}

} // namespace anonymous

CAF_MAIN();
