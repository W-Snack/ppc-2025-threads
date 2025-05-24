#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/fomin_v_conjugate_gradient/include/ops_stl.hpp"

TEST(fomin_v_conjugate_gradient_stl, test_pipeline_run) {
  constexpr int kCount = 990;

  // Создаем трехдиагональную матрицу с диагональным преобладанием
  std::vector<double> input((kCount * kCount) + kCount, 0.0);
  for (int i = 0; i < kCount; ++i) {
    input[(i * kCount) + i] = 4.0;
    if (i > 0) {
      input[(i * kCount) + (i - 1)] = -1.0;
    }
    if (i < kCount - 1) {
      input[(i * kCount) + (i + 1)] = -1.0;
    }
    input[(kCount * kCount) + i] = 1.0;
  }

  // Создаем task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(new double[kCount]));
  task_data_seq->outputs_count.emplace_back(kCount);

  auto test_task_sequential = std::make_shared<fomin_v_conjugate_gradient::FominVConjugateGradientStl>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создаем и инициализируем результаты perf теста
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(fomin_v_conjugate_gradient_stl, test_task_run) {
  constexpr int kCount = 990;

  // Создаем трехдиагональную матрицу с диагональным преобладанием
  std::vector<double> input((kCount * kCount) + kCount, 0.0);
  for (int i = 0; i < kCount; ++i) {
    input[(i * kCount) + i] = 4.0;
    if (i > 0) {
      input[(i * kCount) + (i - 1)] = -1.0;
    }
    if (i < kCount - 1) {
      input[(i * kCount) + (i + 1)] = -1.0;
    }
    input[(kCount * kCount) + i] = 1.0;
  }

  // Создаем task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(new double[kCount]));
  task_data_seq->outputs_count.emplace_back(kCount);

  // Создаем задачу
  auto test_task_sequential = std::make_shared<fomin_v_conjugate_gradient::FominVConjugateGradientStl>(task_data_seq);

  // Создаем атрибуты для perf теста
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создаем и инициализируем результаты perf теста
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Создаем perf анализатор
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}