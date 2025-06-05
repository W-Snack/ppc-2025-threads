#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace leontev_n_fox_stl {

std::vector<double> MatMul(std::vector<double>& a, std::vector<double>& b, size_t n);

class FoxSTL : public ppc::core::Task {
 public:
  explicit FoxSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  [[nodiscard]] double AtA(size_t i, size_t j) const;
  [[nodiscard]] double AtB(size_t i, size_t j) const;
  void MatMulBlocks(size_t a_pos_x, size_t a_pos_y, size_t b_pos_x, size_t b_pos_y, size_t c_pos_x, size_t c_pos_y,
                    size_t size);
  std::vector<double> input_a_;
  std::vector<double> input_b_;
  std::vector<double> output_;
  size_t n_;
};

}  // namespace leontev_n_fox_stl
