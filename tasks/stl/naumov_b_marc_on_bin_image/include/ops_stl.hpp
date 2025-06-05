#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace naumov_b_marc_on_bin_image_stl {

std::vector<int> GenerateRandomBinaryMatrix(int rows, int cols, double probability = 0.5);
std::vector<int> GenerateSparseBinaryMatrix(int rows, int cols, double probability = 0.1);
std::vector<int> GenerateDenseBinaryMatrix(int rows, int cols, double probability = 0.9);

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void ProcessRange(size_t start_idx, size_t end_idx);
  void CreateAndJoinThreads();

  void ProcessPixel(int row, int col);
  void AssignNewLabel(int row, int col);
  void AssignMinLabel(int row, int col, const std::vector<int> &neighbors);

  std::vector<int> FindAdjacentLabels(int row, int col);
  void AssignLabel(int row, int col, int &current_label);
  int FindRoot(int label);
  void UnionLabels(int label1, int label2);

  int rows_{};
  int cols_{};
  std::vector<int> input_image_;
  std::vector<int> output_image_;
  std::vector<int> label_parent_;
  int block_size_ = 64;
  int current_label_ = 0;
};

}  // namespace naumov_b_marc_on_bin_image_stl
