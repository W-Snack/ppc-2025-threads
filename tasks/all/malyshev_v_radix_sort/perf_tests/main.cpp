#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "all/malyshev_v_radix_sort/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

std::vector<double> malyshev_v_radix_sort_all::GetRandomDoubleVector(int size) {
  std::vector<double> vector(size);
  std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
  std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);
  for (int i = 0; i < size; ++i) {
    vector[i] = distribution(generator);
  }
  return vector;
}

TEST(malyshev_v_radix_sort_all, test_pipeline_run) {
  boost::mpi::communicator world;
  const int size = 3000000;
  auto global_vector = malyshev_v_radix_sort_all::GetRandomDoubleVector(size);
  std::vector<double> result(size);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(result.size());

  auto test_task = std::make_shared<malyshev_v_radix_sort_all::TestTaskALL>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 3;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (size_t i = 1; i < result.size(); ++i) {
      ASSERT_LE(result[i - 1], result[i]);
    }
  }
}

TEST(malyshev_v_radix_sort_all, test_task_run) {
  boost::mpi::communicator world;
  const int size = 3000000;
  auto global_vector = malyshev_v_radix_sort_all::GetRandomDoubleVector(size);
  std::vector<double> result(size);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(result.size());

  auto test_task = std::make_shared<malyshev_v_radix_sort_all::TestTaskALL>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 3;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (size_t i = 1; i < result.size(); ++i) {
      ASSERT_LE(result[i - 1], result[i]);
    }
  }
}