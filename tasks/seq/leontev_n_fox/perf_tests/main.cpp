#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/leontev_n_fox/include/ops_seq.hpp"

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

TEST(leontev_n_fox_perf_seq, test_pipeline_run) {
  size_t n = 233;
  std::vector<double> in_data = GenerateRandomMatrix(2 * n * n, 666);
  std::vector<double> out_data(n * n);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.push_back(in_data.size());
  task_data_seq->inputs_count.push_back(1);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(out_data.data()));
  task_data_seq->outputs_count.push_back(out_data.size());

  auto test_task_sequential = std::make_shared<leontev_n_fox_seq::FoxSeq>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(leontev_n_fox_perf_seq, test_task_run) {
  size_t n = 233;
  std::vector<double> in_data = GenerateRandomMatrix(2 * n * n, 666);
  std::vector<double> out_data(n * n);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.push_back(in_data.size());
  task_data_seq->inputs_count.push_back(1);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(out_data.data()));
  task_data_seq->outputs_count.push_back(out_data.size());

  auto test_task_sequential = std::make_shared<leontev_n_fox_seq::FoxSeq>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
