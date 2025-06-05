#include "stl/leontev_n_fox/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace leontev_n_fox_stl {

double FoxSTL::AtA(size_t i, size_t j) const {
  if (i >= n_ || j >= n_) {
    return 0.0;
  }
  return input_a_[(i * n_) + j];
}

double FoxSTL::AtB(size_t i, size_t j) const {
  if (i >= n_ || j >= n_) {
    return 0.0;
  }
  return input_b_[(i * n_) + j];
}

std::vector<double> MatMul(std::vector<double>& a, std::vector<double>& b, size_t n) {
  std::vector<double> res(n * n, 0.0);
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      for (size_t l = 0; l < n; l++) {
        res[(i * n) + j] += a[(i * n) + l] * b[(l * n) + j];
      }
    }
  }
  return res;
}

void FoxSTL::MatMulBlocks(size_t a_pos_x, size_t a_pos_y, size_t b_pos_x, size_t b_pos_y, size_t c_pos_x,
                          size_t c_pos_y, size_t size) {
  size_t row_max = (n_ >= c_pos_y) ? (n_ - c_pos_y) : 0;
  size_t col_max = (n_ >= c_pos_x) ? (n_ - c_pos_x) : 0;
  for (size_t j = 0; j < std::min(size, col_max); j++) {
    for (size_t i = 0; i < std::min(size, row_max); i++) {
      for (size_t l = 0; l < size; l++) {
        output_[((i + c_pos_y) * n_) + (j + c_pos_x)] += AtA(i + a_pos_y, l + a_pos_x) * AtB(l + b_pos_y, j + b_pos_x);
      }
    }
  }
}

bool FoxSTL::PreProcessingImpl() {
  size_t input_count = task_data->inputs_count[0];
  auto* double_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  n_ = reinterpret_cast<size_t*>(task_data->inputs[1])[0];
  input_a_.assign(double_ptr, double_ptr + (input_count / 2));
  input_b_.assign(double_ptr + (input_count / 2), double_ptr + input_count);

  size_t output_count = task_data->outputs_count[0];
  output_.resize(output_count, 0.0);

  return true;
}

bool FoxSTL::ValidationImpl() { return (input_a_.size() == n_ * n_ && output_.size() == n_ * n_); }

bool FoxSTL::RunImpl() {
  size_t div1 = 0;
  const int num_threads = ppc::util::GetPPCNumThreads();
  size_t q = std::min(n_, static_cast<size_t>(std::sqrt(num_threads)));
  if (q == 0) {
    return false;
  }
  size_t k = 0;
  if (n_ % q == 0) {
    k = n_ / q;
  } else {
    k = n_ / q + 1;
  }
  for (size_t l = 0; l < q; l++) {
    std::vector<std::thread> threads;
    threads.reserve(q * q);
    for (size_t i = 0; i < q; i++) {
      for (size_t j = 0; j < q; j++) {
        div1 = ((i + l) % q) * k;
        threads.emplace_back(&FoxSTL::MatMulBlocks, this, div1, i * k, j * k, div1, j * k, i * k, k);
      }
    }
    for (size_t i = 0; i < q * q; i++) {
      threads[i].join();
    }
  }
  return true;
}

bool FoxSTL::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace leontev_n_fox_stl
