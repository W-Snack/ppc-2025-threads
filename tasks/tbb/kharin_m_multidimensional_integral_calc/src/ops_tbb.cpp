#include "tbb/kharin_m_multidimensional_integral_calc/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <cstddef>
#include <functional>
#include <vector>

bool kharin_m_multidimensional_integral_calc_tbb::TestTaskTBB::ValidationImpl() {
  // Проверяем, что предоставлено ровно 3 входа и 1 выход
  if (task_data->inputs.size() != 3 || task_data->outputs.size() != 1) {
    return false;
  }
  // Совпадение grid_sizes и step_sizes
  if (task_data->inputs_count[1] != task_data->inputs_count[2]) {
    return false;
  }
  // Выход должен содержать одно значение
  if (task_data->outputs_count[0] != 1) {
    return false;
  }
  return true;
}

bool kharin_m_multidimensional_integral_calc_tbb::TestTaskTBB::PreProcessingImpl() {
  size_t d = task_data->inputs_count[1];
  auto* sizes_ptr = reinterpret_cast<size_t*>(task_data->inputs[1]);
  grid_sizes_ = std::vector<size_t>(sizes_ptr, sizes_ptr + d);

  // Вычисляем общее количество точек ПОСЛЕДОВАТЕЛЬНО (мало итераций)
  size_t total_size = 1;
  for (size_t i = 0; i < grid_sizes_.size(); i++) {
    total_size *= grid_sizes_[i];
  }

  if (task_data->inputs_count[0] != total_size) {
    return false;
  }
  auto* input_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(input_ptr, input_ptr + total_size);

  if (task_data->inputs_count[2] != d) {
    return false;
  }
  auto* steps_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
  step_sizes_ = std::vector<double>(steps_ptr, steps_ptr + d);

  // Проверяем шаги ПОСЛЕДОВАТЕЛЬНО (мало итераций)
  for (const auto& step : step_sizes_) {
    if (step <= 0.0) {
      return false;
    }
  }

  output_result_ = 0.0;
  return true;
}

bool kharin_m_multidimensional_integral_calc_tbb::TestTaskTBB::RunImpl() {
  // ОСНОВНАЯ ПАРАЛЛЕЛИЗАЦИЯ - здесь много работы!
  double total = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, input_.size()), 0.0,
      [&](const tbb::blocked_range<size_t>& r, double local_sum) -> double {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          local_sum += input_[i];
        }
        return local_sum;
      },
      std::plus<>());

  // Вычисляем элемент объема ПОСЛЕДОВАТЕЛЬНО (мало итераций)
  double volume_element = 1.0;
  for (const auto& step : step_sizes_) {
    volume_element *= step;
  }

  output_result_ = total * volume_element;
  return true;
}

bool kharin_m_multidimensional_integral_calc_tbb::TestTaskTBB::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = output_result_;
  return true;
}