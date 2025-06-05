#include "seq/leontev_n_fox/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace leontev_n_fox_seq {

const int k_ = 4;

double FoxSeq::AtA(size_t i, size_t j) const {
  if (i >= n_ || j >= n_) {
    return 0.0;
  }
  return input_a_[(i * n_) + j];
}

double FoxSeq::AtB(size_t i, size_t j) const {
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

bool FoxSeq::PreProcessingImpl() {
  size_t input_count = task_data->inputs_count[0];
  auto* double_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  n_ = reinterpret_cast<size_t*>(task_data->inputs[1])[0];
  input_a_.assign(double_ptr, double_ptr + (input_count / 2));
  input_b_.assign(double_ptr + (input_count / 2), double_ptr + input_count);
  size_t output_count = task_data->outputs_count[0];
  output_.resize(output_count, 0.0);

  return true;
}

bool FoxSeq::ValidationImpl() {
  return (input_a_.size() == n_ * n_ && input_b_.size() == n_ * n_ && output_.size() == n_ * n_);
}

bool FoxSeq::RunImpl() {
  size_t q = 0;
  size_t div1 = 0;
  if (n_ % k_ == 0) {
    q = n_ / k_;
  } else {
    q = n_ / k_ + 1;
  }
  for (size_t i = 0; i < q; i++) {
    for (size_t j = 0; j < q; j++) {
      for (size_t l = 0; l < q; l++) {
        div1 = ((i + l) % q) * k_;
        // block calc
        for (size_t row = 0; row < std::min(static_cast<size_t>(k_), n_ - (k_ * i)); row++) {
          for (size_t col = 0; col < std::min(static_cast<size_t>(k_), n_ - (k_ * j)); col++) {
            for (size_t m = 0; m < k_; m++) {
              output_[((row + (i * k_)) * n_) + (col + (j * k_))] +=
                  AtA((i * k_) + row, div1 + m) * AtB(div1 + m, (j * k_) + col);
            }
          }
        }
        // block calc end
      }
    }
  }
  return true;
}

bool FoxSeq::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace leontev_n_fox_seq
