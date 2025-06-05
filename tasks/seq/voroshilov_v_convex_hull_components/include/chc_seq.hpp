#pragma once

#include <utility>
#include <vector>

#include "chc.hpp"
#include "core/task/include/task.hpp"

namespace voroshilov_v_convex_hull_components_seq {

class ChcTaskSequential : public ppc::core::Task {
 public:
  explicit ChcTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int height_in_;
  int width_in_;
  std::vector<int> pixels_in_;

  std::vector<Hull> hulls_out_;
};

}  // namespace voroshilov_v_convex_hull_components_seq
