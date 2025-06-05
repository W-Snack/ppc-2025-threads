#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/gromov_a_fox_algorithm/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
void InitializeMatrices(size_t k_n, std::vector<double>& a, std::vector<double>& b, int rank) {
  if (rank == 0) {
    for (size_t i = 0; i < k_n; ++i) {
      for (size_t j = 0; j < k_n; ++j) {
        a[(i * k_n) + j] = static_cast<double>(i + j + 1);
        b[(i * k_n) + j] = static_cast<double>(k_n - i + j + 1);
      }
    }
  }
}

void ComputeExpectedResult(size_t k_n, const std::vector<double>& a, const std::vector<double>& b,
                           std::vector<double>& expected) {
  for (size_t i = 0; i < k_n; ++i) {
    for (size_t j = 0; j < k_n; ++j) {
      for (size_t k = 0; k < k_n; ++k) {
        expected[(i * k_n) + j] += a[(i * k_n) + k] * b[(k * k_n) + j];
      }
    }
  }
}
}  // namespace

TEST(gromov_a_fox_algorithm_all, test_pipeline_run) {
  boost::mpi::communicator world;
  constexpr size_t kN = 400;

  std::vector<double> a(kN * kN, 0.0);
  std::vector<double> b(kN * kN, 0.0);
  std::vector<double> out(kN * kN, 0.0);

  InitializeMatrices(kN, a, b, world.rank());

  std::vector<double> input;
  input.insert(input.end(), a.begin(), a.end());
  input.insert(input.end(), b.begin(), b.end());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_all->inputs_count.emplace_back(input.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  auto test_task_tbb = std::make_shared<gromov_a_fox_algorithm_all::TestTaskAll>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    std::vector<double> expected(kN * kN, 0.0);
    ComputeExpectedResult(kN, a, b, expected);
    for (size_t i = 0; i < out.size(); ++i) {
      EXPECT_NEAR(out[i], expected[i], 1e-3);
    }
  }
}

TEST(gromov_a_fox_algorithm_all, test_task_run) {
  boost::mpi::communicator world;
  constexpr size_t kN = 400;

  std::vector<double> a(kN * kN, 0.0);
  std::vector<double> b(kN * kN, 0.0);
  std::vector<double> out(kN * kN, 0.0);

  InitializeMatrices(kN, a, b, world.rank());

  std::vector<double> input;
  input.insert(input.end(), a.begin(), a.end());
  input.insert(input.end(), b.begin(), b.end());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_all->inputs_count.emplace_back(input.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  auto test_task_tbb = std::make_shared<gromov_a_fox_algorithm_all::TestTaskAll>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    std::vector<double> expected(kN * kN, 0.0);
    ComputeExpectedResult(kN, a, b, expected);
    for (size_t i = 0; i < out.size(); ++i) {
      EXPECT_NEAR(out[i], expected[i], 1e-3);
    }
  }
}