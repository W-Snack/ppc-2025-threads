#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace leontev_n_fox_seq {

std::vector<double> MatMul(std::vector<double>& a, std::vector<double>& b, size_t n);

class FoxSeq : public ppc::core::Task {
 public:
  explicit FoxSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  [[nodiscard]] double AtA(size_t i, size_t j) const;
  [[nodiscard]] double AtB(size_t i, size_t j) const;
  std::vector<double> input_a_;
  std::vector<double> input_b_;
  std::vector<double> output_;
  size_t n_;
};

}  // namespace leontev_n_fox_seq
