#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/leontev_n_fox/include/ops_stl.hpp"

namespace {
std::vector<double> GenerateRandomMatrix(size_t size, int seed, double min_val = 0.0, double max_val = 1.0) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(min_val, max_val);
  std::vector<double> matrix(size);
  for (double& x : matrix) {
    x = dist(rng);
  }
  return matrix;
}
}  // namespace

TEST(leontev_n_fox_stl, 3x3_random) {
  size_t n = 3;
  std::vector<double> in_data = GenerateRandomMatrix(2 * n * n, 666);
  std::vector<double> out_data(n * n);
  std::vector<double> ref_data(n * n);
  std::vector<double> a(n * n);
  std::vector<double> b(n * n);
  std::copy(in_data.begin(), in_data.begin() + static_cast<int>(n * n), a.begin());
  std::copy(in_data.begin() + static_cast<int>(n * n), in_data.begin() + static_cast<int>(2 * n * n), b.begin());
  ref_data = leontev_n_fox_stl::MatMul(a, b, n);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&n));
  task_data->inputs_count.push_back(in_data.size());
  task_data->inputs_count.push_back(1);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_data.data()));
  task_data->outputs_count.push_back(out_data.size());

  leontev_n_fox_stl::FoxSTL task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_NEAR(out_data[(i * n) + j], ref_data[(i * n) + j], 1e-6);
    }
  }
}

TEST(leontev_n_fox_stl, 111x111_random) {
  size_t n = 111;
  std::vector<double> in_data = GenerateRandomMatrix(2 * n * n, 666);
  std::vector<double> out_data(n * n);
  std::vector<double> ref_data(n * n);
  std::vector<double> a(n * n);
  std::vector<double> b(n * n);
  std::copy(in_data.begin(), in_data.begin() + static_cast<int>(n * n), a.begin());
  std::copy(in_data.begin() + static_cast<int>(n * n), in_data.begin() + static_cast<int>(2 * n * n), b.begin());
  ref_data = leontev_n_fox_stl::MatMul(a, b, n);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&n));
  task_data->inputs_count.push_back(in_data.size());
  task_data->inputs_count.push_back(1);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_data.data()));
  task_data->outputs_count.push_back(out_data.size());

  leontev_n_fox_stl::FoxSTL task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_NEAR(out_data[(i * n) + j], ref_data[(i * n) + j], 1e-6);
    }
  }
}

TEST(leontev_n_fox_stl, 5x5) {
  size_t n = 5;
  std::vector<double> in_data = {1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1,
                                 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<double> out_data(n * n);
  std::vector<double> ref_data(n * n);
  std::vector<double> a(n * n);
  std::vector<double> b(n * n);
  std::copy(in_data.begin(), in_data.begin() + static_cast<int>(n * n), a.begin());
  std::copy(in_data.begin() + static_cast<int>(n * n), in_data.begin() + static_cast<int>(2 * n * n), b.begin());
  ref_data = leontev_n_fox_stl::MatMul(a, b, n);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&n));
  task_data->inputs_count.push_back(in_data.size());
  task_data->inputs_count.push_back(1);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_data.data()));
  task_data->outputs_count.push_back(out_data.size());

  leontev_n_fox_stl::FoxSTL task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_NEAR(out_data[(i * n) + j], ref_data[(i * n) + j], 1e-6);
    }
  }
}
